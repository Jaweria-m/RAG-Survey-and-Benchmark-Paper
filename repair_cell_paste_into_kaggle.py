# 13) FIX + RQ4: re-run the LLM judge with robust Yes/No parsing, add an automatic
#     answer-in-context check, compute metric agreement (Table 9) and extra numbers. ~10-15 min.
import re, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

VERIFY = ("Context:\n{ctx}\n\nStatement: {c}\n\n"
          "Is the statement supported by the context above? Answer with one word: Yes or No.")

def parse_yn(t):
    m = re.search(r"\b(yes|no)\b", t.lower())
    return None if m is None else (1.0 if m.group(1) == "yes" else 0.0)

def rejudge(path, n=100):
    rows = rb.read_jsonl(path); todo = rows[:n]
    qs = [r['question'].split('\n\nOptions:')[0] for r in todo]
    ans = [rb._answer_as_text(r) for r in todo]
    claims = [f'The answer to the question "{q}" is: {a}' for q, a in zip(qs, ans)]
    outs = gen([[{'role': 'user', 'content': VERIFY.format(ctx=rb.trunc_words(r['context'], 1500), c=c)}]
                for r, c in zip(todo, claims)], max_tokens=16)
    for r, o in zip(todo, outs):
        r['judge_raw'] = o['text'][:80]
        r['faithfulness'] = parse_yn(o['text'])
        if r['dataset'] == 'hotpotqa':
            a = rb.short_answer(r['pred'])
            r['answer_in_context'] = (None if rb.normalize_answer(a) in ('yes', 'no', '')
                                      else float(rb.has_answer(r['context'], [a])))
    rb._write_jsonl(path, rows)
    return todo

judged = []
for p in sorted(glob.glob(OUT_DIR + '/exp2/*_k5.jsonl')):
    print('re-judging', p); judged += rejudge(p)
df = pd.DataFrame(judged)
print('\nUnparsed judge replies: %.1f%%' % (100 * df['faithfulness'].isna().mean()))
print(df.assign(pred=df['pred'].str[:50])[['dataset', 'config', 'pred', 'judge_raw']].sample(8, random_state=0).to_string())

print('\nTABLE 8 (with corrected faithfulness)')
print(rb.table8(OUT_DIR).to_markdown(index=False))

# ---- Table 9: agreement between automatic metrics (replaces human labels) ----
df['correct'] = np.where(df['dataset'] == 'mirage', df.get('acc'), df.get('f1'))
def rho(sub, a, b):
    s = sub[[a, b]].dropna()
    if len(s) < 10 or s[a].nunique() < 2 or s[b].nunique() < 2:
        return 'n/a'
    r, p = spearmanr(s[a], s[b]); return f"{r:.2f}{'*' if p < 0.05 else ''} (n={len(s)})"
pairs = [('Answer correctness vs LLM-judge faithfulness', 'correct', 'faithfulness'),
         ('Answer correctness vs LLM-judge answer relevancy', 'correct', 'answer_relevancy'),
         ('LLM-judge faithfulness vs answer-in-context check', 'faithfulness', 'answer_in_context'),
         ('Answer correctness vs answer-in-context check', 'correct', 'answer_in_context')]
t9 = pd.DataFrame([{'Metric pair': n, 'General (HotpotQA)': rho(df[df.dataset == 'hotpotqa'], a, b),
                    'Medical (MIRAGE)': rho(df[df.dataset == 'mirage'], a, b), 'All': rho(df, a, b)}
                   for n, a, b in pairs])
print('\nTABLE 9'); print(t9.to_markdown(index=False))
t9.to_csv(OUT_DIR + '/table9_auto.csv', index=False)
h = df[df.dataset == 'hotpotqa'][['faithfulness', 'answer_in_context']].dropna()
if len(h) > 10 and h.nunique().min() > 1:
    print('Cohen kappa, LLM judge vs answer-in-context (HotpotQA): %.2f (n=%d)'
          % (cohen_kappa_score(h.faithfulness.astype(int), h.answer_in_context.astype(int)), len(h)))
print('Answer-in-context rate by architecture (HotpotQA):',
      df[df.dataset == 'hotpotqa'].groupby('config')['answer_in_context'].mean().round(3).to_dict())
print('Judge faithfulness by dataset:', df.groupby('dataset')['faithfulness'].mean().round(3).to_dict())
print('Answer relevancy by dataset:', df.groupby('dataset')['answer_relevancy'].mean().round(3).to_dict())

# ---- extra numbers for Table 10 and the discussion ----
for exp, cfg in (('exp1', 'R0'), ('exp1', 'R4'), ('exp2', 'A1'), ('exp2', 'A3')):
    r = rb.read_jsonl(f'{OUT_DIR}/{exp}/hotpotqa_{cfg}_k5.jsonl')
    print(f'HotpotQA {cfg}: EM = %.1f' % (100 * np.mean([x['em'] for x in r])))
for cfg in ('R0', 'R4'):
    r = pd.DataFrame(rb.read_jsonl(f'{OUT_DIR}/exp1/mirage_{cfg}_k5.jsonl'))
    r['subset'] = r['qid'].str.split('-').str[0]
    print(f'MIRAGE {cfg} accuracy by subset:', (100 * r.groupby('subset')['acc'].mean()).round(1).to_dict(),
          '| unparsed answers:', int(r['pred_choice'].isna().sum()))

import subprocess
subprocess.run('cd /kaggle/working && zip -qr rag_fast.zip rag_fast/exp1 rag_fast/exp2 rag_fast/latency '
               'rag_fast/figures rag_fast/*.json rag_fast/*.csv', shell=True)
print('\nDONE - copy everything printed above and send it to Claude.')
