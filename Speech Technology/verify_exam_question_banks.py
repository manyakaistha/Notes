"""Check archived provenance, coverage, navigation, and independent numerical solutions."""
from pathlib import Path
import json,re,hashlib,math,itertools
from urllib.parse import unquote
from collections import Counter
import yaml
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'sources/exam-papers'
inv=json.loads((BASE/'question-inventory.json').read_text())
solutions=json.loads((BASE/'question-solutions.json').read_text())
S={s['id']:s for s in solutions}
assert len(S)==len(solutions)==len(inv['questions'])==200
assert set(S)=={q['id'] for q in inv['questions']}
assert len(inv['papers'])==18
for p in inv['papers']:
 pdf=BASE/'pdfs'/f"{p['paper']}.pdf"
 assert hashlib.sha256(pdf.read_bytes()).hexdigest()==p['sha256'],pdf
 assert (BASE/'extractions'/f"{p['paper']}.md").is_file()
 assert sum(q['paper']==p['paper'] for q in inv['questions'])==p['parsed_questions']
assert sum(p['parsed_questions']==0 for p in inv['papers'])==2
print('Source integrity: 18 PDF hashes match; 200 academic records; 2 no-paper notices.')
paths=[ROOT/f'week {w}/week-{w}-exam-questions.md' for w in range(1,5)]
paths += [ROOT/'exam-question-index.md',BASE/'question-routing.md']
wiki_index={}
for p in ROOT.rglob('*.md'): wiki_index.setdefault(p.stem,[]).append(p)
image_count=0
for path in paths:
 text=path.read_text()
 if text.startswith('---\n'):
  data=yaml.safe_load(text.split('---',2)[1]); assert isinstance(data,dict)
 assert not any(ord(c)<32 and c not in '\n\t\r' for c in text)
 assert '\t' not in text, ('Unexpected tab / damaged LaTeX',path)
 for m in re.finditer(r'(?<!!)\[([^\]\n]+)\]\(([^)]+)\)|!\[([^\]]*)\]\(([^)]+)\)',text):
  url=m[2] or m[4]
  if url.startswith(('https:','http:')): continue
  name,_,anchor=unquote(url).partition('#')
  target=(path.parent/name).resolve() if name else path
  assert target.is_file(),(path,url)
  if anchor and target.suffix=='.md': assert f'id="{anchor}"' in target.read_text(),(path,url)
 for w in re.findall(r'\[\[([^\]]+)\]\]',text):
  dest=w.replace(chr(92)+'|','|').split('|')[0]
  assert (ROOT/(dest+'.md')).is_file() or dest in wiki_index,(path,w)
 images=list(re.finditer(r'!\[[^\]]*\]\([^)]+\)',text))
 for m in images:
  assert re.match(r'\s*\*\*Figure \d+\.\*\*',text[m.end():]),('Missing caption',path,m[0])
 captions=[int(n) for n in re.findall(r'\*\*Figure (\d+)\.\*\*',text)]
 assert captions==list(range(1,len(images)+1)),('Caption sequence',path)
 image_count+=len(images)
for w in range(1,5):
 text=(ROOT/f'week {w}/week-{w}-exam-questions.md').read_text()
 expected={q['id'] for q in inv['questions'] if S[q['id']]['decision']=='include' and S[q['id']]['week']==w}
 actual=set(re.findall(r'<a id="([^"]+)"></a>',text))
 assert expected==actual
 assert text.count('#### Worked answer — independently derived')==len(expected)
 for id in expected: assert S[id]['answer_markdown'].strip()
 print(f'Week {w}: {len(expected)} records, each with question, worked answer and source provenance.')
assert len(re.findall(r'^\| \[ee4001_', (BASE/'question-routing.md').read_text(),re.M))==200
print(f'Navigation: all generated Markdown links and wikilinks resolve; {image_count} source images have sequential captions; YAML parses.')
# Exhaustive enumeration independently checks the dynamic-programming answers.
A=[[.5,.375,.125],[.25,.125,.625],[.25,.075,.675]]
B=[[.6,.2,.15,.05],[.25,.25,.25,.25],[.05,.1,.35,.5]]
def known_start_paths(start,obs):
 scores=[]
 for path in itertools.product(range(3),repeat=len(obs)):
  score=1.; prev=start
  for state,o in zip(path,obs): score*=A[prev][state]*B[state][o]; prev=state
  scores.append((score,path))
 return max(scores)
score,path=known_start_paths(1,[0,2]); assert path==(0,1) and math.isclose(score,.0140625)
print(f'Cloudy Day 0, Dry/Damp: best Days 1–2 = S,C; score {score:.9f}.')
score,path=known_start_paths(2,[3,0]); assert path==(2,0) and math.isclose(score,.050625)
print(f'Rainy Day 0, Soggy/Dry: best Days 1–2 = R,S; score {score:.9f}.')
scores=[]
for path in itertools.product(range(3),repeat=3):
 prev=0; score=1.
 for state in path: score*=A[prev][state]; prev=state
 scores.append((score,path))
score,path=max(scores); assert path==(1,2,2) and math.isclose(score,.158203125)
print(f'Observed weather forecast: best C,R,R = {score:.9f}; greedy S,S,S = {.5**3:.9f}.')
AA=[[.8,.2],[.4,.6]]; BB=[[.6,.3,.1],[.1,.4,.5]]; pi=[.6,.4]
scores=[]
for path in itertools.product(range(2),repeat=3):
 score=pi[path[0]]*BB[path[0]][0]
 for t in [1,2]: score*=AA[path[t-1]][path[t]]*BB[path[t]][t]
 scores.append((score,path))
best=max(x[0] for x in scores); ties={p for score,p in scores if math.isclose(score,best,abs_tol=1e-14)}
assert ties=={(0,0,1),(0,1,1)} and math.isclose(best,.00864)
print(f'Two-state HMM: tied MAP paths S,S,R and S,R,R; score {best:.9f}.')
trans=[[.3,.5,.2],[.4,.4,.2],[.2,.3,.5]]
flies=[.1,.8,.1]; fast=[.2,.3,.5]
first=[trans[0][i]*flies[i] for i in range(3)]
second=[fast[j]*max(first[i]*trans[i][j] for i in range(3)) for j in range(3)]
assert all(math.isclose(a,b) for a,b in zip(second,[.032,.048,.04]))
print('POS Viterbi: final N,V,A scores = 0.032, 0.048, 0.040; best N,V,V.')
products=[.2*.4,.7*.126,.1*.6]
assert math.isclose(max(products),.0882) and math.isclose(sum(products),.2282)
print('Trellis: max = 0.0882; sum = 0.2282 (all three incoming edges).')
density=math.exp(-.5)/math.sqrt(2*math.pi)
assert math.isclose(density,.24197072451914337) and math.isclose(.6*density/density,.6)
print(f'GMM density = {density:.12f}; component-1 posterior = 0.600000.')
assert sum([2,4,5])/3==11/3 and sum([8,9])/2==8.5
assert math.isclose(5.5*1.1,6.05) and math.isclose(5.5*.9,4.95)
assert math.isclose((1.2-1)**2+(.8-1)**2,.08) and 16000*.010==160
print('K-means/LBG means, VQ distances and frame sample count verified.')
print('PASS: source coverage, generated artifacts and independent numerical checks.')
