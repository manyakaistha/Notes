"""Recompute descriptive past-paper evidence tables for the four exam study guides."""
from pathlib import Path
from collections import Counter
import json,re
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'sources/exam-papers'
Q=json.loads((BASE/'question-inventory.json').read_text())['questions']
S={s['id']:s for s in json.loads((BASE/'question-solutions.json').read_text())}
def family(q,w):
 p=q['paper']; k=q['anchor']
 if w==1:
  if ('Q1' in p and k=='q9' and int(q['term'][:4])<2025) or (p.endswith('2025T2_Q1_NA') and k=='q5') or (p.endswith('2026T1_Q1_NA') and k=='q8'):return 'systems'
  if k=='q10':return 'basis'
  if (k=='q2' and 'Q1' in p) or k=='q14':return 'speech'
  return 'sampling'
 if w==2:
  if 'Q1' in p and int(q['term'][:4])<2025 and k in ['q2','q11','q12']:return 'gaussian'
  if k=='q13' or (p.endswith('2026T1_Q1_NA') and k=='q4'):return 'perception'
  if (p.endswith('2025T2_Q1_NA') and k=='q4') or (p.endswith('2026T1_Q1_NA') and k=='q1') or (p.endswith('2025T2_Q2_NA') and k=='q7'):return 'mfcc'
  return 'stft'
 if w==3:
  if ('Q1' in p and int(q['term'][:4])<2025 and k in ['q4','q5']) or k=='q-mcq-1' or (p.endswith('2025T2_Q1_NA') and k=='q8') or (p.endswith('2026T1_Q1_NA') and k=='q10'):return 'gmm'
  if ('Q1' in p and int(q['term'][:4])<2025 and k=='q6') or k=='q-mcq-8':return 'markov'
  if 'Bayes' in ' '.join(S[q['id']]['topics']):return 'bayes'
  if k=='q-subj-3' or (p.endswith('2023T3_ET_FN') and k=='q-subj-1'):return 'optimization'
  return 'vq'
 if w==4:
  if 'Viterbi' in ' '.join(S[q['id']]['topics']) or ('Q1' in p and k in ['q9','q11']):return 'decoding'
  if 'normalization' in ' '.join(S[q['id']]['topics']) or ('Q2' in p and (p.startswith('ee4001_2022') and k=='q8' or p.startswith('ee4001_2023') and k=='q7')):return 'hmm'
  if k=='q-mcq-1' and p.endswith('2023T3_ET_FN'):return 'forward'
  return 'acoustic'
META={
1:{'sampling':('P1 — practise calculations','Sampling, aliasing and sample counts'),'systems':('P1 — practise proofs','Linearity, time invariance and causality'),'basis':('P2 — rehearse concise answers','Basis / linear-combination idea'),'speech':('P2 — rehearse concise answers','Voicing, articulation and information in speech')},
2:{'gaussian':('P1 — master repeated fundamentals','Gaussian parameters, density normalization and symmetry'),'perception':('P2 — rehearse concise answers','Pitch and basilar membrane'),'stft':('P1 — explain and calculate','STFT, window trade-offs and frame correlation'),'mfcc':('P1 — reconstruct the pipeline','MFCC role and extraction steps')},
3:{'gmm':('P1 — calculate and explain','GMM density, posterior, likelihood and EM'),'vq':('P1 — work iterations by hand','K-means, LBG and vector quantization'),'markov':('P1 — build trellises','Observed Markov forecasts and max-path trellis'),'bayes':('P2 — retrieve accurately','Conditional probability / Bayes rule'),'optimization':('P2 — explain limitations','Local versus global optima')},
4:{'decoding':('P1 — calculate with traceback','HMM Viterbi: weather and POS examples'),'hmm':('P1 — understand every probability','HMM versus chain and probability normalization'),'forward':('P1 — compare with Viterbi','Forward likelihood; foundation for soft alignment'),'acoustic':('P1 — explain and count correctly','HMM–GMM roles, emissions, topology, duration, triphones')}
}
records=[]
for w in range(1,5):
 qq=[q for q in Q if S[q['id']]['decision']=='include' and S[q['id']]['week']==w]
 groups={key:[q for q in qq if family(q,w)==key] for key in META[w]}
 assert sum(map(len,groups.values()))==len(qq)
 table=['| Study priority | Topic family | Question occurrences: Q1 / Q2 / ET | Distinct exam–terms |','| :--- | :--- | :--- | ---: |']
 for key,(priority,label) in META[w].items():
  qs=groups[key]; c=Counter(q['exam'] for q in qs); sittings={(q['term'],q['exam']) for q in qs}
  table.append(f"| {priority} | {label} | {c['Q1']} / {c['Q2']} / {c['ET']} = **{len(qs)}** | {len(sittings)} |")
  records.append(dict(week=w,family=key,label=label,priority=priority,question_ids=[q['id'] for q in qs],occurrences=len(qs),exam_counts=dict(c),distinct_exam_terms=len(sittings)))
 path=ROOT/f'week {w}/week-{w}-exam-focused-notes.md'
 if path.exists():
  text=path.read_text(); text,n=re.subn(r'<!-- evidence:start -->.*?<!-- evidence:end -->','<!-- evidence:start -->\n'+'\n'.join(table)+'\n<!-- evidence:end -->',text,flags=re.S);assert n==1
  path.write_text(text)
 print(f'Week {w}: '+', '.join(f'{k}={len(v)}' for k,v in groups.items()))
(BASE/'exam-study-priorities.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
