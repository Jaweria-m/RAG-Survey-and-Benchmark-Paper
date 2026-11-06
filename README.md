# RAG Survey and Cross-Domain Benchmark

Code and results for the paper:

> **Retrieval-Augmented Generation: A Structured Survey and Cross-Domain Empirical Benchmark of Retrieval Strategies, Architectures, and Evaluation Practices**
> [Author name(s)], [University], 2026.

The benchmark compares retrieval strategies and RAG architectures under one controlled setup: the same generator, prompts and corpora, changing one component at a time. Everything runs on a single free GPU (Kaggle or Colab T4) in about 1.5–2 hours.

## What is compared

| Experiment | Configurations |
| --- | --- |
| 1. Retrieval strategies (RQ2) | R0 no retrieval · R1 BM25 · R2 dense (bge-small-en-v1.5) · R3 hybrid (Reciprocal Rank Fusion) · R4 hybrid + reranker (bge-reranker-base) · R5 R4 + HyDE query rewriting |
| 2. Architectures (RQ3) | A1 naive RAG · A2 extractive compression · A3 IRCoT-style iterative retrieval · A4 CRAG-style corrective retrieval |
| 3. Evaluation agreement (RQ4) | Answer F1/accuracy vs LLM-judge faithfulness and answer relevancy vs a deterministic answer-in-context check |

**Datasets:** HotpotQA (distractor dev set, 200 questions, pooled corpus of 9,735 paragraphs) and MIRAGE (MedQA-US, PubMedQA*, BioASQ-Y/N; 200 questions; MedRAG Textbooks corpus of 125,847 snippets).
**Generator and judge:** Qwen/Qwen2.5-3B-Instruct, greedy decoding. No model is trained or fine-tuned.

## Repository contents

| File | Purpose |
| --- | --- |
| `rag_benchmark_FAST.ipynb` | Main notebook: data loading, indexing, Experiments 1–2, judge, tables, figures |
| `repair_cell_paste_into_kaggle.py` | Final judging step with robust Yes/No parsing, metric-agreement analysis (Table 9) and extra statistics |
| `rag_fast.zip` | All outputs from the reported run (6 November 2026): per-question predictions, retrieved passages, metrics, latencies, figures |

## How to reproduce

1. Open `rag_benchmark_FAST.ipynb` on [Kaggle](https://www.kaggle.com) or [Google Colab](https://colab.research.google.com).
2. Enable a GPU (Kaggle: *Settings → Accelerator → GPU T4 x2*, *Internet → On*; Colab: *Runtime → Change runtime type → T4 GPU*).
3. Paste the contents of `repair_cell_paste_into_kaggle.py` into a new cell at the end of the notebook.
4. Run all cells. Results are written to `rag_fast/`; finished steps are skipped if the session restarts.

All datasets and models download automatically from Hugging Face and GitHub. Question sampling and corpus pooling use seed 42.

## Main results (from the paper)

| | HotpotQA F1 | MIRAGE accuracy |
| --- | --- | --- |
| R0 no retrieval | 24.5 | **62.0** |
| R4 hybrid + rerank | **58.8** | 53.5 |
| A3 iterative | **66.4** | 51.5 |

- Retrieval more than doubles general-domain answer quality, but a general textbook corpus *lowers* accuracy on literature-based medical questions (PubMedQA* 58.2 → 38.8).
- Iterative retrieval improves multi-hop F1 by 7.7 points at 3.4× the latency.
- A 3B LLM judge is unreliable: its faithfulness verdicts show no agreement with a simple grounding check (Cohen's κ = −0.02).

See the paper for confidence intervals, significance tests and the full discussion.

## Environment of the reported run

Kaggle, NVIDIA Tesla T4 (16 GB), Python 3.13.16, PyTorch 2.11.0, CUDA 13.0, Transformers 5.22.0, sentence-transformers 5.7.0, FAISS 1.15.1, bm25s 0.3.16, datasets 4.8.5.

## Citation

```bibtex
@article{ragsurveybenchmark2026,
  title  = {Retrieval-Augmented Generation: A Structured Survey and Cross-Domain Empirical Benchmark of Retrieval Strategies, Architectures, and Evaluation Practices},
  author = {[Author name(s)]},
  year   = {2026},
  note   = {[Journal name, once accepted]}
}
```

## License

MIT. The datasets (HotpotQA, MIRAGE, MedRAG Textbooks) and models keep their original licenses.
