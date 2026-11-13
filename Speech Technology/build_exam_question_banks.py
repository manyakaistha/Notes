"""Rebuild Weeks 1–4 past-paper banks from the archived inventory and reviewed solutions.
Run from this track with: uv run python build_exam_question_banks.py
"""
from pathlib import Path
import json,re,shutil
from collections import Counter
from urllib.parse import quote
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'sources/exam-papers'
INV=json.loads((BASE/'question-inventory.json').read_text())
SOL={s['id']:s for s in json.loads((BASE/'question-solutions.json').read_text())}
PAPERS={p['paper']:p for p in INV['papers']}
assert set(SOL)=={q['id'] for q in INV['questions']}
TOPICS={
1:[('Communication, speech chain and information in waveforms','speech to pure text|emotion'),('Sampling, aliasing and sample counts','sampling|sampled at'),('Quantization and digital amplitude representation',None),('Fourier series, transform taxonomy and DFT/basis representation','basis functions'),('System linearity, invariance and causality','system'),('Convolution, z-transform, poles and zeros',None),('Speech production, phonemes, articulation and voiced/unvoiced/silence','phoneme /f/|voiced sounds')],
2:[('Production/perception: pitch, loudness, basilar membrane and masking','psychophysical|membrane'),('Source–filter model, cepstrum and liftering',None),('STFT, stationarity, frame duration, correlation and spectral resolution','stft|spectrogram|frame|time-windows'),('Mel banks, logarithm, DCT, MFCCs and optional dynamic features','mfcc|mel-frequency cepstral'),('Single-Gaussian review: density, normalization, parameters, symmetry and covariance','gaussian|probability distribution function')],
3:[('Gaussian binary classification, priors, Bayes rule and error analysis','bayes'),('Bivariate Gaussian and covariance geometry',None),('Mixture density, likelihood and component posterior','log-likelihood|1-dimensional mfcc'),('GMM estimation, responsibilities and EM updates','m-step|same cluster centers'),('K-means, LBG, VQ and local/global optimization','k-means|kmeans|linde|local and global|codebook vectors'),('Observed Markov models and transition estimation','weather forcast'),('Greedy versus Viterbi weather forecasts and trellis computation','weather forcast|determine.*alpha')],
4:[('Viterbi max-product, traceback, hard alignment and ties','viterbi|greedy|most likely sequence'),('HMM states, observations, normalization and assumptions','hidden markov|given a hmm'),('Weather–grass HMM worked inference','weather prediction based'),('Forward likelihood and Forward–Backward soft alignment','forward algorithm'),('HMM–GMM emissions, real-valued versus discrete observations','hmm.*gmm|continuous hmm|types of hmm'),('Speech-state topology, self-loop duration, tri-state/triphone and tying','tri-phone|triphone|3-state|types of hmm')]
}
def link(path): return quote(str(path).replace('\\','/'),safe='/#-_.')
def clean(block):
 block=block.split('**Correct answer:**')[0]
 block=re.sub(r'<a id="[^"]+"></a>','',block)
 block=re.sub(r'^\*\*(?:Portal container number|Portal container ID|Question ID|Type|PDF pages|Shared context):\*\*.*$','',block,flags=re.M)
 block=re.sub(r'^\*Source option ID:\*.*$','',block,flags=re.M)
 return re.sub(r'\n{3,}','\n\n',block).strip()
def identity(q):
 sec='MCQ' if 'mcq' in q['anchor'] else 'Subjective' if 'subj' in q['anchor'] else q['type'].split(' · ')[0]
 return q['title']+' ('+sec+')'
def target(q): return f"week {SOL[q['id']]['week']}/week-{SOL[q['id']]['week']}-exam-questions.md#{q['id']}"
def title(p):
 year=p['term'][:4]; term=p['term'][5:]; exam={'Q1':'Quiz 1','Q2':'Quiz 2','ET':'End Term'}[p['exam']]
 return f"{year} Term {term} · {exam} · {p['paper'].split('_')[-1]}"
def selected(week): return [q for q in INV['questions'] if SOL[q['id']]['decision']=='include' and SOL[q['id']]['week']==week]
counts={w:len(selected(w)) for w in range(1,5)}
for week in range(1,5):
 qs=selected(week); folder=ROOT/f'week {week}'; figure=[0]
 def render(block):
  def image(m):
   source=BASE/'extractions'/m[2]
   assert source.is_file(),source
   rel=Path('assets/exam-papers')/Path(m[2]).relative_to('assets')
   out=folder/rel; out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,out)
   return f'![{m[1]}]({link(rel)})'
  block=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',image,clean(block))
  def caption(m):
   figure[0]+=1; return f'**Figure {figure[0]}.** '+m[1]
  block=re.sub(r'\*\*Figure \d+\.\*\*\s*([^\n]*)',caption,block)
  return block
 out=[f'''---
title: "Week {week}: Authentic Exam Questions and Worked Answers"
week: {week}
date: 2026-11-10
status: complete
tags: [speech/week-0{week}, speech/past-papers]
map: "[[Knowledge-Map]]"
parent: "[[week-{week}-notes]]"
---

# Week {week} — Authentic Exam Questions and Worked Answers

**Navigation:** [[week-{week}-notes|Study notes]] · [[week-{week}-exam-focused-notes|Exam-focused learning guide]] · [[exam-question-index|All weeks and source audit]]

**Coverage:** {len(qs)} source question occurrences. Repeated questions are retained for each paper/session. Questions come from Q1, Q2 and ET whenever their tested content falls in this week's confirmed scope. These are authentic paper extractions, separate from generated practice in the study notes.

**Answer provenance:** The supplied PDFs do not provide verifiable academic answer keys. All solutions below are independently derived, with source ambiguities stated explicitly. Read the question and source figure before checking the worked answer. Physical PDF page numbers include cover/portal pages. FN/AN/NA are retained exactly from filenames; NA means no session specified in the filename.

## Topic checklist and evidence

The topic list follows the confirmed weekly notes. “No direct question found” describes these supplied papers, not permission to skip the topic. A question can support several rows.

| Topic | Matching source questions |
| :--- | :--- |''']
 for topic,pattern in TOPICS[week]:
  matches=[q for q in qs if pattern and re.search(pattern,q['raw_block'],re.I)]
  refs=', '.join(f"[{q['paper'].replace('ee4001_','')} {q['anchor']}](#{q['id']})" for q in matches) or 'No direct question found; revise the linked study notes.'
  out.append(f'| {topic} | {refs} |')
 out.append('\n## Question navigator\n\n| Paper | Question | Topic | Answer status |\n| :--- | :--- | :--- | :--- |')
 for q in qs:
  s=SOL[q['id']]; out.append(f"| {q['paper'].replace('ee4001_','')} | [{identity(q)}](#{q['id']}) | {'; '.join(s['topics'])} | {s['answer_status']} |")
 for exam in ['Q1','Q2','ET']:
  group=[q for q in qs if q['exam']==exam]
  if not group: continue
  out.append('\n## '+{'Q1':'Quiz 1','Q2':'Quiz 2','ET':'End Term'}[exam]+' questions\n')
  for q in group:
   s=SOL[q['id']]; p=PAPERS[q['paper']]; page=re.search(r'\d+',str(q['pages']))
   out.append(f"\n<a id=\"{q['id']}\"></a>\n\n### {title(p)} — {identity(q)}\n\n**Source:** [Original PDF, p. {q['pages']}](../sources/exam-papers/pdfs/{q['paper']}.pdf#page={page[0] if page else 1}) · [Archived extraction](../sources/exam-papers/extractions/{q['paper']}.md#{q['anchor']})  \n**Topics:** {'; '.join(s['topics'])}. **Marks:** {q['marks'] or 'not stated'}.\n")
   if s['secondary_weeks']: out.append('**Overlap:** '+', '.join(f'[[week-{w}-notes|Week {w}]]' for w in s['secondary_weeks'])+'. Primary placement follows the earliest substantive mechanism.\n')
   if s['included_subparts']!='all': out.append(f"**Partial scope:** Included: {s['included_subparts']}. Beyond Weeks 1–4: {s['excluded_subparts']}. The full question is retained as context; this is not a complete answer to the later-topic question.\n")
   for ctx in q['context_ids']:
    c=p['contexts'][ctx]
    if 'Section Heading' not in c['block']: out.append('**Shared source instructions:**\n\n'+render(c['block'])+'\n')
   out.append(render(q['raw_block']))
   out.append('\n#### Worked answer — independently derived\n\n'+s['answer_markdown']+'\n')
   if s['answer_status']=='needs-review': out.append('> [!warning] Source ambiguity\n> The answer above identifies the precise defect or missing assumption. No official key was available to settle it.\n')
   out.append('\n---\n')
 out.append('\n## Reproducibility and source integrity\n\nSource PDFs, original extracted text and crops are archived under `sources/exam-papers/`. The inventory records PDF SHA-256 hashes. Rebuild with `uv run python build_exam_question_banks.py`; verify numerical examples and generated links with `uv run python verify_exam_question_banks.py`. The reviewed routing manifest includes every source question, including excluded later material.\n')
 (folder/f'week-{week}-exam-questions.md').write_text('\n'.join(out))
 # Link from the master note exactly once.
 note=folder/f'week-{week}-notes.md'; body=note.read_text(); banner=f'**Authentic past papers:** [[week-{week}-exam-questions|Week {week} questions and worked answers]] · [[exam-question-index|Paper coverage and routing audit]]\n\n'
 if f'[[week-{week}-exam-questions' not in body:
  pos=re.search(r'^# .+\n',body,re.M).end(); body=body[:pos]+'\n'+banner+body[pos:]; note.write_text(body)
# Index and complete routing audit.
idx=['''---
title: "Speech Technology: Exam Question Index — Weeks 1–4"
date: 2026-11-10
tags: [speech/past-papers, speech/moc]
map: "[[Knowledge-Map]]"
---

# Exam Question Index — Weeks 1–4

[[Knowledge-Map|Knowledge Map]] · [[exam-study-guide|Exam-focused learning route]]

All 18 supplied PDF files were inventoried, containing 200 academic question records. Two files are administrative no-paper notices and contain zero academic questions. Questions are routed by the confirmed topic scope, including early topics appearing in later exams. Repeated paper occurrences remain separate; these counts are not counts of unique problem templates.

**Exam rules supplied by the learner:** Quiz 1 covers Weeks 1–4; Quiz 2 covers Weeks 1–8 with emphasis on Weeks 5–8; End Term covers Weeks 1–12 with emphasis on Weeks 9–12. `2026T1` means 2026, Term 1. FN/AN/NA session labels are preserved. Excluded later questions are archived without guessed week numbers for the unconfirmed Weeks 6–12 syllabus.

**Solutions:** independently derived, not official keys. Ambiguous options, missing GMM parameters, and OCR/notation discrepancies are explained beside the affected questions. One mixed Transformer-ASR question contributes only its explicitly requested STFT front-end portion.

## Weekly banks

| Week | Bank | Question occurrences |
| :--- | :--- | ---: |''']
for w in range(1,5): idx.append(f'| {w} | [[week-{w}-exam-questions\|Questions and worked answers]] | {counts[w]} |')
idx.append(f"\n**Total included:** {sum(counts.values())}; **excluded:** {200-sum(counts.values())}. See [complete routing audit](sources/exam-papers/question-routing.md) for all 200 decisions, including exclusions.\n\n## Paper inventory\n\n| Paper / session | Academic questions | W1 | W2 | W3 | W4 | Excluded | Status |\n| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
for p in INV['papers']:
 qq=[q for q in INV['questions'] if q['paper']==p['paper']]; c=Counter(SOL[q['id']]['week'] for q in qq if SOL[q['id']]['decision']=='include')
 idx.append(f"| [{p['paper']}](sources/exam-papers/pdfs/{p['paper']}.pdf) | {len(qq)} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {len(qq)-sum(c.values())} | {'No academic paper: administrative notice' if not qq else 'Inventoried and routed'} |")
idx.append('\n## Items requiring source caution\n')
for q in INV['questions']:
 s=SOL[q['id']]
 if s['decision']=='include' and s['answer_status']!='derived': idx.append(f"- [{q['paper']} — {identity(q)}]({link(target(q))}): {s['answer_status']}; {'; '.join(s['topics'])}.")
idx.append('''
## Review and rebuild

Luna agents performed the text/scope/solution pass; Terra inspected targeted scanned figures. Parent review corrected omissions, source variants, numerical calculations and ambiguous options. All raw source archives remain unchanged.

- [Inventory and source hashes](sources/exam-papers/question-inventory.json)
- [Reviewed answers and routing data](sources/exam-papers/question-solutions.json)
- [Targeted source-figure audit](sources/exam-papers/visual-audit.json)
- [Numerical and file verification output](sources/exam-papers/verification-output.txt)

Run from the Speech Technology track:

```sh
uv run python build_exam_question_banks.py
uv run python verify_exam_question_banks.py
```
''')
(ROOT/'exam-question-index.md').write_text('\n'.join(idx))
routing=['# Complete question-routing audit\n\n[[exam-question-index|Exam index]] · [[Knowledge-Map]]\n\nAll academic records are listed once. Included questions have one primary week; genuine overlaps are marked in their banks. Excluded questions retain their exact source in the archive.\n\n| Source record | Decision | Primary week | Rationale / included portion |\n| :--- | :--- | :--- | :--- |']
for q in INV['questions']:
 s=SOL[q['id']]; url='../../'+link(target(q)) if s['decision']=='include' else f'extractions/{q["paper"]}.md#{q["anchor"]}'
 rationale=s['reason']+(f" Included: {s['included_subparts']}." if s['included_subparts'] not in ('all','none') else '')
 rationale=rationale.replace(chr(124), chr(92)+chr(124))
 routing.append(f"| [{q['id']}]({url}) | {s['decision']} | {s['week'] or 'Outside current scope'} | {rationale} |")
(BASE/'question-routing.md').write_text('\n'.join(routing)+'\n')
km=ROOT/'Knowledge-Map.md'; text=km.read_text()
if '[[exam-question-index' not in text: text+='\n## Authentic exam questions\n\n[[exam-question-index|Weeks 1–4 paper inventory, questions and worked answers]]\n\n'+ '\n'.join(f'- [[week-{w}-exam-questions|Week {w}: {counts[w]} source question occurrences]]' for w in range(1,5))+'\n'
km.write_text(text)
print('Generated banks: '+', '.join(f'W{w}={n}' for w,n in counts.items())+f'; {sum(counts.values())} included / {len(INV["questions"])} total records.')
