"""rag_bench.py - controlled RAG benchmark for the paper
"Retrieval-Augmented Generation: A Systematic Survey and Cross-Domain Empirical
Benchmark of Retrieval Strategies, Architectures, and Evaluation Practices".

Experiment 1 (RQ2): retrieval configurations R0-R5
Experiment 2 (RQ3): architectures A1-A4 (+ helpers for optional A5 LightRAG)
Part B (RQ4): LLM-judge faithfulness / answer relevancy
Part C (RQ4): human-annotation export, agreement, statistics, tables, figures

Every function writes JSONL results to OUT_DIR so a disconnected session can resume.
"""
import os, re, json, gzip, time, math, random, string, collections, urllib.request, glob

# Colab/Kaggle ship JAX; bm25s uses it automatically, and JAX grabs 75% of GPU memory on first use,
# which leaves no room for the generator (CUDA out of memory). Keep JAX on the CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

# =============================================================================
# 1. Text normalisation and answer metrics
# =============================================================================
_PUNCT = set(string.punctuation)


def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred, golds):
    p = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        common = collections.Counter(p) & collections.Counter(gt)
        ns = sum(common.values())
        if ns == 0 or not p or not gt:
            continue
        prec, rec = ns / len(p), ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def has_answer(text, golds):
    t = " " + normalize_answer(text) + " "
    for g in golds:
        ng = normalize_answer(g)
        if ng and (" " + ng + " ") in t:
            return True
    return False


def short_answer(text):
    """Take the first line and strip a leading 'Answer:' so EM is fair."""
    t = text.strip().split("\n")[0].strip()
    t = re.sub(r"^(final\s+)?answer\s*[:\-]\s*", "", t, flags=re.I)
    t = re.sub(r"^so the answer is\s*[:\-]?\s*", "", t, flags=re.I)
    return t.strip().rstrip(".")


def parse_choice(text, letters):
    """Extract an option letter (e.g. 'B') from model output; None if not found."""
    t = text.strip()
    letters = [l.upper() for l in letters]
    if t and t[0] in letters and (len(t) == 1 or not t[1].isalpha()):
        return t[0]
    m = re.search(r"answer\s*(?:is|:)\s*\(?([A-Za-z])\b", t, flags=re.I)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    m = re.search(r"\(([A-Za-z])\)", t)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    for m in re.finditer(r"\b([A-Z])\b", t):
        if m.group(1) in letters:
            return m.group(1)
    return None


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def trunc_words(text, n=300):
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " ..."


# =============================================================================
# 2. Dataset loaders. Each returns a dict:
#    name, task ('short' | 'mcq' | 'legal'), corpus (list of docs), items (eval list)
#    doc  = {id, title, text, [file, start, end]}
# =============================================================================

def _rng_sample(lst, n, seed):
    idx = list(range(len(lst)))
    random.Random(seed).shuffle(idx)
    return [lst[i] for i in idx[:n]]


def load_hotpotqa(n_eval=500, pool_q=5000, seed=42):
    """HotpotQA distractor dev set. Corpus = all paragraphs of `pool_q` questions
    (gold + distractors, de-duplicated by title); eval questions come from the pool."""
    from datasets import load_dataset
    try:
        dsr = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    except Exception:
        dsr = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    rows = _rng_sample(list(range(len(dsr))), min(pool_q, len(dsr)), seed)
    corpus, seen, items = [], set(), []
    for j, ri in enumerate(rows):
        r = dsr[ri]
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            if title not in seen:
                seen.add(title)
                corpus.append({"id": title, "title": title, "text": "".join(sents).strip()})
        if j < n_eval:
            items.append({"qid": r["id"], "question": r["question"], "answers": [r["answer"]],
                          "gold_ids": sorted(set(r["supporting_facts"]["title"])), "type": r["type"]})
    return {"name": "hotpotqa", "task": "short", "corpus": corpus, "items": items}


DPR_NQ_DEV = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"


def load_nq(cache_dir, n_eval=500, pool_q=5000, seed=42, url=DPR_NQ_DEV):
    """Natural Questions dev (DPR release). Corpus = positive + negative + hard-negative
    passages of `pool_q` questions (de-duplicated). Relevance = passage contains a gold answer."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "biencoder-nq-dev.json.gz")
    if not os.path.exists(path):
        print("Downloading DPR NQ dev ...")
        urllib.request.urlretrieve(url, path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = [d for d in json.load(f) if d.get("positive_ctxs")]
    rows = _rng_sample(data, min(pool_q, len(data)), seed)
    corpus, seen, items = [], set(), []
    for j, d in enumerate(rows):
        ctxs = d["positive_ctxs"][:3] + d.get("hard_negative_ctxs", [])[:10] + d.get("negative_ctxs", [])[:10]
        for c in ctxs:
            pid = str(c.get("passage_id") or (c["title"] + "|" + c["text"][:50]))
            if pid not in seen:
                seen.add(pid)
                corpus.append({"id": pid, "title": c["title"], "text": c["text"]})
        if j < n_eval:
            items.append({"qid": f"nq{j}", "question": d["question"], "answers": d["answers"]})
    return {"name": "nq", "task": "short", "corpus": corpus, "items": items}


MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"


def load_mirage(cache_dir, n_eval=500, seed=42, subsets=("medqa", "pubmedqa", "bioasq"),
                corpus_name="MedRAG/textbooks", url=MIRAGE_URL):
    """MIRAGE (MedQA-US, PubMedQA*, BioASQ-Y/N), stratified sample; corpus = MedRAG Textbooks."""
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "mirage_benchmark.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    bench = json.load(open(path, encoding="utf-8"))
    per = [n_eval // len(subsets) + (1 if i < n_eval % len(subsets) else 0) for i in range(len(subsets))]
    items = []
    for sub, n in zip(subsets, per):
        qs = bench[sub]
        for qid in _rng_sample(sorted(qs), min(n, len(qs)), seed):
            q = qs[qid]
            opts = q["options"]
            ans = q["answer"]
            if ans not in opts:  # answer given as text
                ans = next((k for k, v in opts.items() if str(v).strip().lower() == str(ans).strip().lower()), ans)
            items.append({"qid": f"{sub}-{qid}", "subset": sub, "question": q["question"],
                          "options": opts, "answer": ans})
    c = load_dataset(corpus_name, split="train")
    corpus = [{"id": str(r["id"]), "title": r.get("title", ""), "text": r["content"]} for r in c]
    return {"name": "mirage", "task": "mcq", "corpus": corpus, "items": items}


def _merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def load_legalbench_rag(legal_dir, n_eval=500, seed=42, chunk_chars=2000, overlap=250):
    """LegalBench-RAG. `legal_dir` must contain corpus/ and benchmarks/ from the official
    download (see the LegalBench-RAG GitHub README). Documents are chunked into
    `chunk_chars`-character windows with `overlap` characters of overlap."""
    bench_files = sorted(glob.glob(os.path.join(legal_dir, "benchmarks", "*.json")))
    if not bench_files:
        raise FileNotFoundError("No benchmarks/*.json found in " + legal_dir)
    per = [n_eval // len(bench_files) + (1 if i < n_eval % len(bench_files) else 0) for i in range(len(bench_files))]
    texts, items = {}, []

    def doc_text(fp):
        if fp not in texts:
            texts[fp] = open(os.path.join(legal_dir, "corpus", fp), encoding="utf-8", errors="ignore").read()
        return texts[fp]

    for bf, n in zip(bench_files, per):
        tests = json.load(open(bf, encoding="utf-8"))["tests"]
        name = os.path.splitext(os.path.basename(bf))[0]
        for i, t in enumerate(_rng_sample(tests, min(n, len(tests)), seed)):
            spans = [(s["file_path"], int(s["span"][0]), int(s["span"][1])) for s in t["snippets"]]
            gold_text = [doc_text(fp)[a:b] for fp, a, b in spans]
            items.append({"qid": f"{name}-{i}", "subset": name, "question": t["query"],
                          "gold_spans": spans, "answers": [" ".join(gold_text)]})
    # chunk every corpus file
    corpus = []
    files = sorted(glob.glob(os.path.join(legal_dir, "corpus", "**", "*.txt"), recursive=True))
    step = chunk_chars - overlap
    for full in files:
        fp = os.path.relpath(full, os.path.join(legal_dir, "corpus")).replace(os.sep, "/")
        txt = doc_text(fp)
        for s in range(0, max(1, len(txt)), step):
            e = min(len(txt), s + chunk_chars)
            corpus.append({"id": f"{fp}#{s}", "title": os.path.basename(fp), "text": txt[s:e],
                           "file": fp, "start": s, "end": e})
            if e >= len(txt):
                break
    return {"name": "legalbench_rag", "task": "legal", "corpus": corpus, "items": items}


# =============================================================================
# 3. Retrieval relevance and metrics
# =============================================================================

def _legal_overlap(doc, spans):
    return sum(max(0, min(doc["end"], b) - max(doc["start"], a)) for fp, a, b in spans if fp == doc["file"])


def is_relevant(ds, item, doc):
    n = ds["name"]
    if n == "hotpotqa":
        return doc["id"] in item["gold_ids"]
    if n == "nq":
        return has_answer(doc["text"], item["answers"])
    if n == "legalbench_rag":
        return _legal_overlap(doc, item["gold_spans"]) > 0
    return None  # mirage: no gold passages


def n_relevant(ds, item):
    """Number of relevant documents in the whole corpus (used for recall and ideal DCG).
    HotpotQA: gold supporting paragraphs. LegalBench-RAG: chunks overlapping a gold span
    (counted once per item and cached). NQ: 1 (answer-bearing passages are not enumerated)."""
    if ds["name"] == "hotpotqa":
        return len(item["gold_ids"])
    if ds["name"] == "legalbench_rag":
        if "_n_rel" not in item:
            if "_by_file" not in ds:
                ds["_by_file"] = collections.defaultdict(list)
                for d in ds["corpus"]:
                    ds["_by_file"][d["file"]].append(d)
            files = {fp for fp, _, _ in item["gold_spans"]}
            item["_n_rel"] = max(1, sum(1 for fp in files for d in ds["_by_file"][fp]
                                        if _legal_overlap(d, item["gold_spans"]) > 0))
        return item["_n_rel"]
    return 1


def legal_char_prf(item, docs):
    """Character-level precision / recall / F1 of retrieved chunks vs gold spans (LegalBench-RAG)."""
    gold = collections.defaultdict(list)
    for fp, a, b in item["gold_spans"]:
        gold[fp].append((a, b))
    ret = collections.defaultdict(list)
    for d in docs:
        ret[d["file"]].append((d["start"], d["end"]))
    g_tot = sum(e - s for fp in gold for s, e in _merge(gold[fp]))
    r_tot = sum(e - s for fp in ret for s, e in _merge(ret[fp]))
    inter = 0
    for fp in gold:
        for gs, ge in _merge(gold[fp]):
            for rs, re_ in _merge(ret.get(fp, [])):
                inter += max(0, min(ge, re_) - max(gs, rs))
    p = inter / r_tot if r_tot else 0.0
    r = inter / g_tot if g_tot else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def retrieval_metrics(ds, item, docs, ks=(1, 3, 5, 10)):
    """docs = ranked list (>=10) of doc dicts. Returns {} for datasets without gold passages."""
    if ds["name"] == "mirage" or not docs:
        return {}
    rel = [bool(is_relevant(ds, item, d)) for d in docs]
    nrel = n_relevant(ds, item)
    out = {}
    for k in ks:
        if ds["name"] == "legalbench_rag":
            p, r, f = legal_char_prf(item, docs[:k])
            out[f"recall@{k}"], out[f"precision@{k}"], out[f"retF1@{k}"] = r, p, f
        else:
            out[f"recall@{k}"] = min(sum(rel[:k]), nrel) / nrel
    first = next((i for i, x in enumerate(rel) if x), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(rel[:10]) if x)
    # NQ: the full set of answer-bearing passages is unknown, so the ideal ranking uses
    # max(1, relevant passages found in the top 10); this keeps nDCG within [0, 1].
    n_ideal = max(nrel, sum(rel[:10])) if ds["name"] == "nq" else nrel
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_ideal, 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def answer_metrics(ds, item, raw):
    if ds["task"] == "mcq":
        choice = parse_choice(raw, list(item["options"].keys()))
        return {"acc": float(choice == item["answer"]), "pred_choice": choice}
    if ds["task"] == "legal":
        return {"f1": token_f1(raw, item["answers"])}
    pred = short_answer(raw)
    return {"em": exact_match(pred, item["answers"]), "f1": token_f1(pred, item["answers"])}


# =============================================================================
# 4. Model wrappers (real models on GPU)
# =============================================================================

def load_encoder(model_id="BAAI/bge-m3", max_len=512, device="cuda"):
    """Returns encode(list[str]) -> L2-normalised float32 matrix."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device=device)
    m.max_seq_length = max_len
    if device == "cuda":
        m.half()

    def encode(texts, batch_size=64, show=False):
        return m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=show).astype("float32")
    return encode


def load_reranker(model_id="BAAI/bge-reranker-v2-m3", max_len=512, device="cuda"):
    """Returns score(list[(query, passage)]) -> np.array of relevance probabilities in [0, 1]."""
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_id, max_length=max_len, device=device)
    if device == "cuda":
        ce.model.half()

    def score(pairs, batch_size=64):
        if not pairs:
            return np.zeros(0)
        s = np.asarray(ce.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype="float32")
        if s.min() < 0 or s.max() > 1:
            s = 1.0 / (1.0 + np.exp(-s))
        return s
    return score


class VLLMGenerator:
    """Greedy generation with vLLM. __call__(list_of_message_lists) -> list of
    {'text', 'in_tok', 'out_tok'} (or a list of texts per prompt when n > 1)."""

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=8192,
                 gpu_memory_utilization=0.6, seed=0):
        import torch
        from vllm import LLM
        dtype = "half" if torch.cuda.get_device_capability()[0] < 8 else "auto"
        self.model_id = model_id
        self.llm = LLM(model=model_id, max_model_len=max_model_len, dtype=dtype,
                       gpu_memory_utilization=gpu_memory_utilization, seed=seed, enforce_eager=False)
        self.tok = self.llm.get_tokenizer()
        self.extra = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}

    def __call__(self, messages_list, max_tokens=256, temperature=0.0, n=1, quiet=False):
        from vllm import SamplingParams
        if not messages_list:
            return []
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **self.extra)
                   for m in messages_list]
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, seed=0)
        outs = self.llm.generate(prompts, sp, use_tqdm=not quiet)
        if n > 1:
            return [[o.text.strip() for o in out.outputs] for out in outs]
        return [{"text": o.outputs[0].text.strip(), "in_tok": len(o.prompt_token_ids),
                 "out_tok": len(o.outputs[0].token_ids)} for o in outs]


class HFGenerator:
    """Plain Hugging Face transformers generator (no vLLM). Same interface as VLLMGenerator.
    Batched greedy decoding; n > 1 uses sampling (for answer relevancy)."""

    def __init__(self, model_id="Qwen/Qwen2.5-3B-Instruct", batch_size=8, max_input_tokens=4096):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.model_id, self.bs, self.max_in = model_id, batch_size, max_input_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dt = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.extra = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}

    def __call__(self, messages_list, max_tokens=256, temperature=0.0, n=1, quiet=False):
        torch = self.torch
        if not messages_list:
            return []
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **self.extra)
                   for m in messages_list]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        results = [None] * len(prompts)
        bs = max(1, self.bs // n)
        it = range(0, len(order), bs)
        if not quiet and len(prompts) > bs:
            try:
                from tqdm.auto import tqdm
                it = tqdm(it, desc="generate", leave=False)
            except ImportError:
                pass
        for s in it:
            idx = order[s:s + bs]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_in, add_special_tokens=False).to(self.model.device)
            kw = dict(max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id, num_return_sequences=n)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                out = self.model.generate(**enc, **kw)
            new = out[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(new, skip_special_tokens=True)
            in_lens = enc["attention_mask"].sum(dim=1).tolist()
            for j, i in enumerate(idx):
                if n > 1:
                    results[i] = [t.strip() for t in texts[j * n:(j + 1) * n]]
                else:
                    out_len = int((new[j] != self.tok.pad_token_id).sum())
                    results[i] = {"text": texts[j].strip(), "in_tok": int(in_lens[j]), "out_tok": out_len}
        return results


# =============================================================================
# 5. Retrieval stack: BM25, dense, hybrid (RRF), rerank, HyDE
# =============================================================================

def rrf(rank_lists, top_n, c=60):
    scores = collections.defaultdict(float)
    for lst in rank_lists:
        for r, i in enumerate(lst):
            scores[int(i)] += 1.0 / (c + r + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


def doc_string(d, n_words=300):
    return (d["title"] + ". " if d.get("title") else "") + trunc_words(d["text"], n_words)


class RetrievalStack:
    def __init__(self, ds, encoder, reranker, cache_dir, bm25_k1=0.9, bm25_b=0.4):
        import bm25s, Stemmer
        self.ds, self.corpus = ds, ds["corpus"]
        self.encoder, self.reranker = encoder, reranker
        self.timings = {}
        texts = [doc_string(d) for d in self.corpus]
        self._stemmer = Stemmer.Stemmer("english")
        t0 = time.time()
        self.bm = bm25s.BM25(k1=bm25_k1, b=bm25_b)
        self.bm.index(bm25s.tokenize(texts, stopwords="en", stemmer=self._stemmer, show_progress=False),
                      show_progress=False)
        self.timings["bm25_index_s"] = time.time() - t0
        os.makedirs(cache_dir, exist_ok=True)
        emb_path = os.path.join(cache_dir, f"{ds['name']}_emb_{len(self.corpus)}.npy")
        t0 = time.time()
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
        else:
            emb = encoder(texts, show=True) if _accepts_show(encoder) else encoder(texts)
            np.save(emb_path, emb)
        self.timings["dense_index_s"] = time.time() - t0
        import faiss
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    # each retrieval method returns a list (per query) of corpus indices, best first
    def bm25(self, queries, n):
        import bm25s
        n = min(n, len(self.corpus))
        qt = bm25s.tokenize(list(queries), stopwords="en", stemmer=self._stemmer, show_progress=False)
        try:
            res, _ = self.bm.retrieve(qt, k=n, show_progress=False, backend_selection="numpy")
        except TypeError:
            res, _ = self.bm.retrieve(qt, k=n, show_progress=False)
        return [list(map(int, r)) for r in res]

    def dense(self, queries, n):
        q = self.encoder(list(queries))
        _, I = self.index.search(q, min(n, len(self.corpus)))
        return [[int(i) for i in row if i >= 0] for row in I]

    def hybrid(self, queries, n, pool=100):
        b, d = self.bm25(queries, pool), self.dense(queries, pool)
        return [rrf([bi, di], n) for bi, di in zip(b, d)]

    def rerank(self, queries, cand_lists, n):
        pairs, owners = [], []
        for qi, (q, cands) in enumerate(zip(queries, cand_lists)):
            for c in cands:
                pairs.append((q, doc_string(self.corpus[c], 350)))
                owners.append((qi, c))
        s = self.reranker(pairs)
        per = collections.defaultdict(list)
        for (qi, c), sc in zip(owners, s):
            per[qi].append((float(sc), c))
        ranked, scores = [], []
        for qi in range(len(queries)):
            lst = sorted(per[qi], key=lambda x: -x[0])[:n]
            ranked.append([c for _, c in lst])
            scores.append([sc for sc, _ in lst])
        return ranked, scores

    def run_config(self, cfg, queries, gen=None, n_out=10):
        """Returns (ranked_lists, extra) for R1-R5. extra holds HyDE texts / token costs."""
        extra = {"in_tok": [0] * len(queries), "out_tok": [0] * len(queries)}
        if cfg == "R1":
            return self.bm25(queries, n_out), extra
        if cfg == "R2":
            return self.dense(queries, n_out), extra
        if cfg == "R3":
            return self.hybrid(queries, n_out), extra
        if cfg == "R4":
            ranked, sc = self.rerank(queries, self.hybrid(queries, 50), n_out)
            extra["scores"] = sc
            return ranked, extra
        if cfg == "R5":
            outs = gen([[{"role": "user", "content": HYDE_PROMPT.format(q=q)}] for q in queries], max_tokens=128)
            hyps = [o["text"] for o in outs]
            extra["in_tok"] = [o["in_tok"] for o in outs]
            extra["out_tok"] = [o["out_tok"] for o in outs]
            extra["hyde"] = hyps
            d = self.dense(hyps, 100)
            b = self.bm25([q + " " + h for q, h in zip(queries, hyps)], 100)
            fused = [rrf([bi, di], 50) for bi, di in zip(b, d)]
            ranked, sc = self.rerank(queries, fused, n_out)
            extra["scores"] = sc
            return ranked, extra
        raise ValueError(cfg)


def _accepts_show(fn):
    try:
        import inspect
        return "show" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# =============================================================================
# 6. Prompts
# =============================================================================
SYSTEM = "You are a careful assistant that answers questions accurately."
HYDE_PROMPT = ("Write a short factual passage (about 80 words) that answers the question below. "
               "Do not mention that it is hypothetical.\n\nQuestion: {q}\nPassage:")
REWRITE_PROMPT = ("Rewrite the question below as a short web-search query containing only the key "
                  "terms. Output only the query.\n\nQuestion: {q}\nQuery:")
INSTR = {
    "short": "Answer the question with a short answer only (an entity, a number, a date, or yes/no). Do not explain.",
    "mcq": "Choose the single correct option. Reply with the option letter only.",
    "legal": "Answer the question. Quote the exact sentence(s) from the excerpts that answer it, as briefly as possible.",
}


def question_text(ds, item):
    if ds["task"] == "mcq":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        return f"{item['question']}\n\nOptions:\n{opts}"
    return item["question"]


def format_context(docs, n_words=300):
    return "\n\n".join(f"[{i + 1}] {doc_string(d, n_words)}" for i, d in enumerate(docs))


def build_messages(ds, item, docs, reasoning=None):
    parts = []
    if docs:
        parts.append("Documents:\n" + format_context(docs))
    if reasoning:
        parts.append("Reasoning so far:\n" + reasoning)
    parts.append(INSTR[ds["task"]])
    parts.append(f"Question: {question_text(ds, item)}\nAnswer:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# =============================================================================
# 7. Experiment 1 (RQ2): retrieval configurations
# =============================================================================
EXP1_CONFIGS = ("R0", "R1", "R2", "R3", "R4", "R5")


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _gold_for_row(ds, it):
    if ds["task"] == "mcq":
        return it["answer"]
    return it["answers"][0][:1000]


def run_exp1(ds, stack, gen, out_dir, configs=EXP1_CONFIGS, k=5, max_tokens=256):
    items, corpus = ds["items"], ds["corpus"]
    qs = [it["question"] for it in items]
    for cfg in configs:
        path = os.path.join(out_dir, "exp1", f"{ds['name']}_{cfg}_k{k}.jsonl")
        if os.path.exists(path):
            print("skip (done):", path)
            continue
        print(f"== Exp1 {ds['name']} {cfg} k={k}")
        t0 = time.time()
        if cfg == "R0":
            ranked, extra = [[] for _ in qs], {"in_tok": [0] * len(qs), "out_tok": [0] * len(qs)}
        else:
            ranked, extra = stack.run_config(cfg, qs, gen, n_out=max(10, k))
        t_ret = time.time() - t0
        msgs = [build_messages(ds, it, [corpus[i] for i in r[:k]]) for it, r in zip(items, ranked)]
        t1 = time.time()
        outs = gen(msgs, max_tokens=max_tokens)
        t_gen = time.time() - t1
        rows = []
        for j, (it, r, o) in enumerate(zip(items, ranked, outs)):
            docs10 = [corpus[i] for i in r[:10]]
            row = {"qid": it["qid"], "dataset": ds["name"], "exp": "exp1", "config": cfg, "k": k,
                   "question": question_text(ds, it), "gold": _gold_for_row(ds, it), "pred": o["text"],
                   "context": format_context(docs10[:k]) if docs10 else "",
                   "retrieved": [corpus[i]["id"] for i in r[:10]],
                   "in_tok": o["in_tok"] + extra["in_tok"][j], "out_tok": o["out_tok"] + extra["out_tok"][j],
                   "n_retrieval": 0 if cfg == "R0" else 1,
                   "batch_time_per_q": (t_ret + t_gen) / len(items)}
            row.update(retrieval_metrics(ds, it, docs10))
            row.update(answer_metrics(ds, it, o["text"]))
            rows.append(row)
        _write_jsonl(path, rows)
        print(f"   saved {path}  ({(t_ret + t_gen) / 60:.1f} min)")


# =============================================================================
# 8. Experiment 2 (RQ3): architectures A1-A4
# =============================================================================

def _compress(stack, q, doc_ids, keep=8):
    """Extractive compression: rerank the sentences of the retrieved docs, keep the top `keep`
    in their original order (RECOMP-style extractive compressor)."""
    sents = []
    for rank, di in enumerate(doc_ids):
        for pos, s in enumerate(split_sentences(stack.corpus[di]["text"])):
            sents.append((rank, pos, s))
    if not sents:
        return " ".join(stack.corpus[d]["text"] for d in doc_ids)
    sc = stack.reranker([(q, s) for _, _, s in sents])
    top = sorted(np.argsort(-sc)[:keep])
    return " ".join(sents[i][2] for i in top)


def arch_naive(ds, stack, gen, items, k, retr_cfg):
    qs = [it["question"] for it in items]
    ranked, extra = stack.run_config(retr_cfg, qs, gen, n_out=max(10, k))
    docs = [[stack.corpus[i] for i in r[:k]] for r in ranked]
    outs = gen([build_messages(ds, it, d) for it, d in zip(items, docs)], quiet=len(items) == 1)
    return [{"pred": o["text"], "ranked": r, "context": format_context(d),
             "in_tok": o["in_tok"] + extra["in_tok"][j], "out_tok": o["out_tok"] + extra["out_tok"][j],
             "n_retrieval": 1} for j, (o, r, d) in enumerate(zip(outs, ranked, docs))]


def arch_advanced(ds, stack, gen, items, k, retr_cfg, keep=8):
    qs = [it["question"] for it in items]
    ranked, extra = stack.run_config(retr_cfg, qs, gen, n_out=max(10, k))
    ctx = [_compress(stack, q, r[:k], keep) for q, r in zip(qs, ranked)]
    docs = [[{"title": "Relevant excerpts", "text": c}] for c in ctx]
    outs = gen([build_messages(ds, it, d) for it, d in zip(items, docs)], quiet=len(items) == 1)
    return [{"pred": o["text"], "ranked": r, "context": c,
             "in_tok": o["in_tok"] + extra["in_tok"][j], "out_tok": o["out_tok"] + extra["out_tok"][j],
             "n_retrieval": 1} for j, (o, r, c) in enumerate(zip(outs, ranked, ctx))]


IRCOT_STEP = ("Documents:\n{ctx}\n\nQuestion: {q}\n\nReasoning so far: {r}\n\n"
              "Write the NEXT reasoning step as ONE short sentence, using the documents. "
              "If you can already answer, write: So the answer is: <answer>.")


def arch_iterative(ds, stack, gen, items, k, retr_cfg, max_steps=3, max_docs=15):
    """IRCoT-style: interleave one reasoning sentence with one retrieval, up to `max_steps` rounds."""
    qs = [it["question"] for it in items]
    ranked, extra = stack.run_config(retr_cfg, qs, gen, n_out=max(10, k))
    pools = [list(r[:k]) for r in ranked]
    first_ranked = [list(r) for r in ranked]
    thoughts, done = [""] * len(items), [False] * len(items)
    calls = [1] * len(items)
    in_tok = list(extra["in_tok"])
    out_tok = list(extra["out_tok"])
    for _ in range(max_steps):
        active = [j for j in range(len(items)) if not done[j]]
        if not active:
            break
        msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": IRCOT_STEP.format(
            ctx=format_context([stack.corpus[i] for i in pools[j]], 150), q=question_text(ds, items[j]),
            r=thoughts[j] or "(none)")}] for j in active]
        outs = gen(msgs, max_tokens=64, quiet=len(items) == 1)
        new_q, new_j = [], []
        for j, o in zip(active, outs):
            in_tok[j] += o["in_tok"]
            out_tok[j] += o["out_tok"]
            sent = (split_sentences(o["text"]) or [o["text"].strip()])[0]
            thoughts[j] = (thoughts[j] + " " + sent).strip()
            if "so the answer is" in o["text"].lower():
                done[j] = True
            else:
                new_q.append(sent)
                new_j.append(j)
        if new_q:
            more, ex2 = stack.run_config(retr_cfg, new_q, gen, n_out=k)
            for idx, (j, r) in enumerate(zip(new_j, more)):
                calls[j] += 1
                in_tok[j] += ex2["in_tok"][idx]
                out_tok[j] += ex2["out_tok"][idx]
                for d in r:
                    if d not in pools[j] and len(pools[j]) < max_docs:
                        pools[j].append(d)
    msgs = [build_messages(ds, it, [stack.corpus[i] for i in p], reasoning=t)
            for it, p, t in zip(items, pools, thoughts)]
    outs = gen(msgs, quiet=len(items) == 1)
    res = []
    for j, o in enumerate(outs):
        res.append({"pred": o["text"], "ranked": first_ranked[j], "pool": pools[j], "reasoning": thoughts[j],
                    "context": format_context([stack.corpus[i] for i in pools[j]]),
                    "in_tok": in_tok[j] + o["in_tok"], "out_tok": out_tok[j] + o["out_tok"],
                    "n_retrieval": calls[j]})
    return res


def arch_corrective(ds, stack, gen, items, k, retr_cfg=None, upper=0.5, lower=0.1, keep=8):
    """CRAG-style: the reranker acts as retrieval evaluator. Best score >= upper -> Correct
    (use docs); < lower -> Incorrect (rewrite query and re-retrieve); otherwise Ambiguous
    (merge both). Then knowledge refinement by sentence-level filtering."""
    qs = [it["question"] for it in items]
    ranked, sc = stack.rerank(qs, stack.hybrid(qs, 50), max(10, k))
    action = ["correct" if (s and s[0] >= upper) else "incorrect" if (not s or s[0] < lower) else "ambiguous"
              for s in sc]
    need = [j for j, a in enumerate(action) if a != "correct"]
    in_tok, out_tok, calls = [0] * len(items), [0] * len(items), [1] * len(items)
    final = [list(r[:k]) for r in ranked]
    if need:
        rw = gen([[{"role": "user", "content": REWRITE_PROMPT.format(q=qs[j])}] for j in need],
                 max_tokens=48, quiet=len(items) == 1)
        rq = [o["text"].split("\n")[0] for o in rw]
        r2, s2 = stack.rerank([qs[j] for j in need], stack.hybrid(rq, 50), k)
        for idx, j in enumerate(need):
            in_tok[j] += rw[idx]["in_tok"]
            out_tok[j] += rw[idx]["out_tok"]
            calls[j] += 1
            if action[j] == "incorrect":
                final[j] = r2[idx]
            else:
                merged = sorted(set(zip(sc[j][:k], ranked[j][:k])) | set(zip(s2[idx], r2[idx])), key=lambda x: -x[0])
                seen, lst = set(), []
                for s, d in merged:
                    if d not in seen:
                        seen.add(d)
                        lst.append(d)
                final[j] = lst[:k]
    ctx = [_compress(stack, q, f, keep) for q, f in zip(qs, final)]
    docs = [[{"title": "Relevant excerpts", "text": c}] for c in ctx]
    outs = gen([build_messages(ds, it, d) for it, d in zip(items, docs)], quiet=len(items) == 1)
    return [{"pred": o["text"], "ranked": ranked[j], "final_docs": final[j], "action": action[j],
             "context": ctx[j], "in_tok": in_tok[j] + o["in_tok"], "out_tok": out_tok[j] + o["out_tok"],
             "n_retrieval": calls[j]} for j, o in enumerate(outs)]


ARCHS = {"A1": arch_naive, "A2": arch_advanced, "A3": arch_iterative, "A4": arch_corrective}


def run_exp2(ds, stack, gen, out_dir, retr_cfg="R4", archs=("A1", "A2", "A3", "A4"), k=5,
             latency_n=50, subset=None):
    """retr_cfg = best retrieval configuration from Experiment 1 (e.g. 'R4').
    latency_n = number of questions also run one-at-a-time to measure per-query latency."""
    items = ds["items"] if subset is None else ds["items"][:subset]
    corpus = ds["corpus"]
    for a in archs:
        path = os.path.join(out_dir, "exp2", f"{ds['name']}_{a}_k{k}.jsonl")
        if os.path.exists(path):
            print("skip (done):", path)
            continue
        print(f"== Exp2 {ds['name']} {a} (retriever {retr_cfg})")
        fn = ARCHS[a]
        t0 = time.time()
        res = fn(ds, stack, gen, items, k, retr_cfg)
        batch_t = (time.time() - t0) / len(items)
        lat = []
        for it in items[:latency_n]:
            t1 = time.time()
            fn(ds, stack, gen, [it], k, retr_cfg)
            lat.append(time.time() - t1)
        rows = []
        for it, r in zip(items, res):
            docs10 = [corpus[i] for i in r["ranked"][:10]]
            row = {"qid": it["qid"], "dataset": ds["name"], "exp": "exp2", "config": a, "k": k,
                   "retriever": retr_cfg, "question": question_text(ds, it), "gold": _gold_for_row(ds, it),
                   "pred": r["pred"], "context": r["context"], "in_tok": r["in_tok"], "out_tok": r["out_tok"],
                   "n_retrieval": r["n_retrieval"], "batch_time_per_q": batch_t}
            for extra_key in ("action", "reasoning"):
                if extra_key in r:
                    row[extra_key] = r[extra_key]
            row.update(retrieval_metrics(ds, it, docs10))
            row.update(answer_metrics(ds, it, r["pred"]))
            rows.append(row)
        _write_jsonl(path, rows)
        _write_jsonl(os.path.join(out_dir, "latency", f"{ds['name']}_{a}.jsonl"),
                     [{"dataset": ds["name"], "config": a, "latency_s": x} for x in lat])
        print(f"   saved {path}; median latency {np.median(lat) if lat else float('nan'):.2f}s")


# ---- optional A5 (graph RAG) helpers -------------------------------------------------

def export_a5_subset(ds, out_dir, n=200):
    """Writes the first n questions and the docs needed for them so LightRAG can index a small
    corpus. Run A1 on the same subset (run_exp2(..., archs=('A1',), subset=n)) for a fair comparison."""
    items = ds["items"][:n]
    os.makedirs(os.path.join(out_dir, "a5"), exist_ok=True)
    _write_jsonl(os.path.join(out_dir, "a5", f"{ds['name']}_questions.jsonl"),
                 [{"qid": it["qid"], "question": question_text(ds, it)} for it in items])
    return items


def score_a5(ds, out_dir, answers_path, n=200):
    """answers_path: JSONL with {qid, pred, context, in_tok, out_tok, latency_s} produced by your LightRAG run."""
    by_id = {it["qid"]: it for it in ds["items"][:n]}
    rows = []
    for a in read_jsonl(answers_path):
        it = by_id[a["qid"]]
        row = {"qid": a["qid"], "dataset": ds["name"], "exp": "exp2", "config": "A5", "k": 5,
               "question": question_text(ds, it), "gold": _gold_for_row(ds, it), "pred": a["pred"],
               "context": a.get("context", ""), "in_tok": a.get("in_tok", 0), "out_tok": a.get("out_tok", 0),
               "n_retrieval": a.get("n_retrieval", 1)}
        row.update(answer_metrics(ds, it, a["pred"]))
        rows.append(row)
    _write_jsonl(os.path.join(out_dir, "exp2", f"{ds['name']}_A5_k5.jsonl"), rows)
    lat = [{"dataset": ds["name"], "config": "A5", "latency_s": a["latency_s"]}
           for a in read_jsonl(answers_path) if "latency_s" in a]
    if lat:
        _write_jsonl(os.path.join(out_dir, "latency", f"{ds['name']}_A5.jsonl"), lat)


# =============================================================================
# 9. Part B (RQ4): LLM-judge faithfulness and answer relevancy (RAGAS-style)
# =============================================================================
CLAIMS_PROMPT = ("Break the answer below into a list of short, self-contained factual claims "
                 "(at most 5). Output one claim per line and nothing else.\n\nQuestion: {q}\nAnswer: {a}")
VERIFY_PROMPT = ("Context:\n{ctx}\n\nStatement: {c}\n\nCan the statement be directly inferred from the "
                 "context above? Reply with Yes or No only.")
GENQ_PROMPT = ("Write one question that the following answer responds to. Output only the question.\n\nAnswer: {a}")


def _answer_as_text(row):
    pred = row["pred"].strip()
    m = re.search(r"Options:\n(.*)$", row["question"], flags=re.S)
    if m:  # MCQ: turn the letter into the option text
        opts = dict(re.findall(r"^([A-Z])\. (.*)$", m.group(1), flags=re.M))
        ch = parse_choice(pred, list(opts.keys()))
        return opts.get(ch, pred)
    return pred


def judge_rows(rows, judge, encoder, n_ctx_words=1500):
    """Adds 'faithfulness' (share of answer claims supported by the context, Es et al. 2024)
    and 'answer_relevancy' (mean cosine similarity between the original question and 3 questions
    generated from the answer) to each row. Rows with empty context get faithfulness = None."""
    qs_plain = [r["question"].split("\n\nOptions:")[0] for r in rows]
    ans = [_answer_as_text(r) for r in rows]
    # 1) claims: short answers become one claim; long (legal) answers are decomposed by the judge
    claims = [None] * len(rows)
    long_idx = [i for i, r in enumerate(rows) if r["dataset"] == "legalbench_rag"]
    for i in range(len(rows)):
        if i not in set(long_idx):
            claims[i] = [f'The answer to the question "{qs_plain[i]}" is: {ans[i]}']
    if long_idx:
        outs = judge([[{"role": "user", "content": CLAIMS_PROMPT.format(q=qs_plain[i], a=ans[i])}] for i in long_idx],
                     max_tokens=256)
        for i, o in zip(long_idx, outs):
            cl = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in o["text"].split("\n")]
            claims[i] = [c for c in cl if c][:5] or [ans[i]]
    # 2) verify every claim against the context actually shown to the generator
    pairs = [(i, c) for i, r in enumerate(rows) if r.get("context") for c in claims[i]]
    verdicts = judge([[{"role": "user", "content": VERIFY_PROMPT.format(
        ctx=trunc_words(rows[i]["context"], n_ctx_words), c=c)}] for i, c in pairs], max_tokens=4)
    sup = collections.defaultdict(list)
    for (i, _), v in zip(pairs, verdicts):
        sup[i].append(1.0 if v["text"].strip().lower().startswith("yes") else 0.0)
    # 3) answer relevancy
    gq = judge([[{"role": "user", "content": GENQ_PROMPT.format(a=a)}] for a in ans],
               max_tokens=64, temperature=0.7, n=3)
    q_emb = encoder(qs_plain)
    flat = [x for lst in gq for x in lst]
    g_emb = encoder(flat) if flat else np.zeros((0, q_emb.shape[1]))
    pos = 0
    for i, r in enumerate(rows):
        n = len(gq[i])
        sims = g_emb[pos:pos + n] @ q_emb[i]
        pos += n
        r["answer_relevancy"] = float(np.mean(sims)) if n else None
        r["faithfulness"] = float(np.mean(sup[i])) if sup.get(i) else None
    return rows


def run_judge(out_dir, judge, encoder, n_per_file=None, patterns=("exp1/*_k5.jsonl", "exp2/*_k5.jsonl")):
    """Scores every k=5 result file (skips R0, which has no context). n_per_file limits cost."""
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(out_dir, pat))):
            rows = read_jsonl(path)
            if not rows or rows[0]["config"] == "R0" or "faithfulness" in rows[0]:
                continue
            todo = rows if n_per_file is None else rows[:n_per_file]
            print("judging", path, len(todo))
            judge_rows(todo, judge, encoder)
            _write_jsonl(path, rows)


# =============================================================================
# 10. Part C: statistics, tables, figures, human evaluation
# =============================================================================

def bootstrap_ci(vals, n_boot=10000, seed=0, alpha=0.05):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Mean difference a-b with 95% CI and two-sided p-value (paired by question)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975)), "p": float(min(1.0, p))}


def mcnemar_test(a, b):
    """a, b = 0/1 correctness of two systems on the same questions."""
    from statsmodels.stats.contingency_tables import mcnemar
    a, b = np.asarray(a, int), np.asarray(b, int)
    tbl = [[int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())],
           [int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())]]
    r = mcnemar(tbl, exact=True)
    return {"table": tbl, "statistic": float(r.statistic), "p": float(r.pvalue)}


def compare_configs(out_dir, exp, dataset, cfg_a, cfg_b, metric, k=5):
    ra = {r["qid"]: r for r in read_jsonl(os.path.join(out_dir, exp, f"{dataset}_{cfg_a}_k{k}.jsonl"))}
    rb = {r["qid"]: r for r in read_jsonl(os.path.join(out_dir, exp, f"{dataset}_{cfg_b}_k{k}.jsonl"))}
    ids = [q for q in ra if q in rb]
    a, b = [ra[q][metric] for q in ids], [rb[q][metric] for q in ids]
    out = {"n": len(ids), **paired_bootstrap(a, b)}
    if metric == "acc":
        out["mcnemar"] = mcnemar_test(a, b)
    return out


def _fmt(vals, scale=100.0, digits=1):
    m, lo, hi = bootstrap_ci(vals)
    if math.isnan(m):
        return "–"
    return f"{m * scale:.{digits}f} ± {(hi - lo) / 2 * scale:.{digits}f}"


def load_results(out_dir, exp, k=5):
    res = {}
    for p in glob.glob(os.path.join(out_dir, exp, f"*_k{k}.jsonl")):
        rows = read_jsonl(p)
        if rows:
            res[(rows[0]["dataset"], rows[0]["config"])] = rows
    return res


def _col(rows, key):
    if not rows or key not in rows[0]:
        return None
    return [r.get(key) for r in rows]


def table7(out_dir, k=5):
    """Paper Table 7: retrieval quality and answer accuracy (mean ± half-width of 95% bootstrap CI, %)."""
    import pandas as pd
    res = load_results(out_dir, "exp1", k)
    spec = [("NQ R@5", "nq", f"recall@{k}"), ("NQ EM", "nq", "em"),
            ("HotpotQA R@5", "hotpotqa", f"recall@{k}"), ("HotpotQA F1", "hotpotqa", "f1"),
            ("MIRAGE Acc.", "mirage", "acc"),
            ("LegalBench-RAG R@5", "legalbench_rag", f"recall@{k}"), ("LegalBench-RAG F1", "legalbench_rag", "f1")]
    names = {"R0": "R0 Closed-book", "R1": "R1 BM25", "R2": "R2 Dense", "R3": "R3 Hybrid",
             "R4": "R4 Hybrid + rerank", "R5": "R5 R4 + query rewrite"}
    data = []
    for cfg in EXP1_CONFIGS:
        row = {"Config": names[cfg]}
        for col, dsn, key in spec:
            vals = _col(res.get((dsn, cfg)), key)
            row[col] = _fmt(vals) if vals else "–"
        data.append(row)
    return _drop_empty(pd.DataFrame(data))


def _drop_empty(df):
    """Drop columns for datasets that were not run (all cells '\u2013')."""
    keep = [c for c in df.columns if (df[c].astype(str) != "\u2013").any()]
    return df[keep]


def table_appendix_retrieval(out_dir, k=5):
    """MRR and nDCG@10 per dataset and configuration (for an appendix table)."""
    import pandas as pd
    res = load_results(out_dir, "exp1", k)
    data = []
    for (dsn, cfg), rows in sorted(res.items()):
        if "mrr" in rows[0]:
            data.append({"Dataset": dsn, "Config": cfg, "MRR": _fmt(_col(rows, "mrr")),
                         "nDCG@10": _fmt(_col(rows, "ndcg@10"))})
    return pd.DataFrame(data)


def table8(out_dir, k=5):
    """Paper Table 8: accuracy, faithfulness, efficiency per architecture."""
    import pandas as pd
    res = load_results(out_dir, "exp2", k)
    lat = collections.defaultdict(list)
    for p in glob.glob(os.path.join(out_dir, "latency", "*.jsonl")):
        for r in read_jsonl(p):
            lat[r["config"]].append(r["latency_s"])
    names = {"A1": "A1 Naive", "A2": "A2 Advanced", "A3": "A3 Iterative", "A4": "A4 Corrective",
             "A5": "A5 Graph-based*"}
    data = []
    for a in ("A1", "A2", "A3", "A4", "A5"):
        if not any(c == a for _, c in res):
            continue
        row = {"Architecture": names[a]}
        for col, dsn, key in [("NQ EM", "nq", "em"), ("HotpotQA F1", "hotpotqa", "f1"),
                              ("MIRAGE Acc.", "mirage", "acc"), ("Legal F1", "legalbench_rag", "f1")]:
            vals = _col(res.get((dsn, a)), key)
            row[col] = _fmt(vals) if vals else "–"
        allrows = [r for (d, c), rs in res.items() if c == a for r in rs]
        f = [r.get("faithfulness") for r in allrows if r.get("faithfulness") is not None]
        row["Faithfulness"] = _fmt(f) if f else "[run judge]"
        row["Latency (s)"] = f"{np.median(lat[a]):.2f}" if lat[a] else "–"
        row["Tokens / query"] = f"{np.mean([r['in_tok'] + r['out_tok'] for r in allrows]):.0f}"
        row["Retrieval calls / query"] = f"{np.mean([r['n_retrieval'] for r in allrows]):.2f}"
        data.append(row)
    return _drop_empty(pd.DataFrame(data))


DOMAIN = {"nq": "General", "hotpotqa": "General", "mirage": "Medical", "legalbench_rag": "Legal"}


def export_for_annotation(out_dir, n_per_domain=50, seed=7):
    """Stratified sample of judged answers (any config except R0) -> CSV for two annotators."""
    import pandas as pd
    pool = collections.defaultdict(list)
    for exp in ("exp1", "exp2"):
        for (dsn, cfg), rows in load_results(out_dir, exp).items():
            if cfg == "R0":
                continue
            for r in rows:
                if r.get("faithfulness") is not None:
                    pool[DOMAIN[dsn]].append(r)
    rng = random.Random(seed)
    out = []
    for dom in ("General", "Medical", "Legal"):
        rows = pool[dom]
        rng.shuffle(rows)
        for r in rows[:n_per_domain]:
            out.append({"sample_id": f"{r['dataset']}|{r['config']}|{r['qid']}", "domain": dom,
                        "question": r["question"], "gold": r["gold"], "context": r["context"][:4000],
                        "answer": r["pred"],
                        "a1_correct": "", "a1_faithful": "", "a2_correct": "", "a2_faithful": "",
                        "final_correct": "", "final_faithful": "",
                        "em": r.get("em", r.get("acc")), "f1": r.get("f1", r.get("acc")),
                        "llm_faithfulness": r.get("faithfulness"), "llm_relevancy": r.get("answer_relevancy")})
    df = pd.DataFrame(out)
    path = os.path.join(out_dir, "human_annotation.csv")
    df.to_csv(path, index=False)
    print("wrote", path, len(df), "rows. Fill a1_*/a2_* with 0 or 1; fill final_* only where they disagree.")
    return path


def analyze_annotation(csv_path):
    """Returns (kappa dict, Table 9 DataFrame of Spearman rho with significance stars)."""
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score
    df = pd.read_csv(csv_path)
    kappa = {}
    for lab in ("correct", "faithful"):
        a1, a2 = df[f"a1_{lab}"].astype(int), df[f"a2_{lab}"].astype(int)
        kappa[lab] = float(cohen_kappa_score(a1, a2))
        final = df[f"final_{lab}"]
        df[f"human_{lab}"] = np.where(a1 == a2, a1, pd.to_numeric(final, errors="coerce"))
    if df[["human_correct", "human_faithful"]].isna().any().any():
        print("WARNING: some disagreements have no final_* label; they are dropped.")
    spec = [("Exact Match vs human correctness", "em", "human_correct"),
            ("Token F1 vs human correctness", "f1", "human_correct"),
            ("LLM-judge answer relevancy vs human correctness", "llm_relevancy", "human_correct"),
            ("LLM-judge faithfulness vs human faithfulness", "llm_faithfulness", "human_faithful")]
    rows = []
    doms = [d for d in ("General", "Medical", "Legal") if (df.domain == d).any()] + ["All"]
    for name, m, h in spec:
        row = {"Metric": name}
        for dom in doms:
            sub = df if dom == "All" else df[df.domain == dom]
            sub = sub[[m, h]].dropna()
            if dom == "Legal" and m == "em":
                row[dom] = "–"
                continue
            if len(sub) < 5 or sub[m].nunique() < 2 or sub[h].nunique() < 2:
                row[dom] = "n/a"
                continue
            rho, p = spearmanr(sub[m], sub[h])
            row[dom] = f"{rho:.2f}{'*' if p < 0.05 else ''}"
        rows.append(row)
    return kappa, pd.DataFrame(rows)


def make_figures(out_dir, k=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    res = load_results(out_dir, "exp1", k)
    metrics = [("NQ (EM)", "nq", "em"), ("HotpotQA (F1)", "hotpotqa", "f1"),
               ("MIRAGE (Acc.)", "mirage", "acc"), ("LegalBench-RAG (F1)", "legalbench_rag", "f1")]
    metrics = [m for m in metrics if any((m[1], c) in res for c in EXP1_CONFIGS)]
    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(metrics)), 4.2))
    w = 0.13
    for ci, cfg in enumerate(EXP1_CONFIGS):
        ms, lo_e, hi_e = [], [], []
        for _, dsn, key in metrics:
            vals = _col(res.get((dsn, cfg)), key)
            m, lo, hi = bootstrap_ci(vals) if vals else (np.nan, np.nan, np.nan)
            ms.append(m * 100)
            lo_e.append((m - lo) * 100)
            hi_e.append((hi - m) * 100)
        x = np.arange(len(metrics)) + (ci - 2.5) * w
        ax.bar(x, ms, w, yerr=[lo_e, hi_e], capsize=2, label=cfg)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Score (%)")
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig5_retrieval.png"), dpi=300)
    plt.close(fig)
    # Fig 6: accuracy vs latency
    res2 = load_results(out_dir, "exp2", k)
    lat = collections.defaultdict(list)
    for p in glob.glob(os.path.join(out_dir, "latency", "*.jsonl")):
        for r in read_jsonl(p):
            lat[r["config"]].append(r["latency_s"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for a in ("A1", "A2", "A3", "A4", "A5"):
        scores = []
        for _, dsn, key in metrics:
            vals = _col(res2.get((dsn, a)), key)
            if vals:
                scores.append(np.mean(vals) * 100)
        if scores and lat[a]:
            x, y = np.median(lat[a]), np.mean(scores)
            ax.scatter(x, y, s=50)
            ax.annotate(a, (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xscale("log")
    ax.set_title("A5 averaged only over the datasets it was run on", fontsize=8, loc="left")
    ax.set_xlabel("Median latency per query (s, log scale)")
    ax.set_ylabel("Mean score across datasets (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig6_accuracy_latency.png"), dpi=300)
    plt.close(fig)
    print("figures saved in", figdir)


def heatmap_table9(table9_df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [c for c in ("General", "Medical", "Legal", "All") if c in table9_df.columns]
    M = np.array([[float(str(v).rstrip("*")) if str(v).rstrip("*").replace(".", "", 1).lstrip("-").isdigit()
                   else np.nan for v in table9_df[c]] for c in cols]).T
    fig, ax = plt.subplots(figsize=(6.5, 3))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(table9_df)))
    ax.set_yticklabels(table9_df["Metric"], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    path = os.path.join(out_dir, "figures", "fig7_metric_agreement.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def dataset_stats(ds):
    return {"dataset": ds["name"], "eval_questions": len(ds["items"]), "corpus_docs": len(ds["corpus"])}
