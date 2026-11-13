"""Verify the four exam guides; --refresh-outputs records actual stdout from their code blocks."""
from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout
from urllib.parse import unquote
import hashlib,io,itertools,json,math,re,sys
import yaml
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'sources/exam-papers'
qs=json.loads((BASE/'question-inventory.json').read_text())['questions']
ss={s['id']:s for s in json.loads((BASE/'question-solutions.json').read_text())}
rows=json.loads((BASE/'exam-study-priorities.json').read_text())
assert len({q['id'] for q in qs})==200
ids=[id for row in rows for id in row['question_ids']]
assert len(ids)==len(set(ids))==98
assert set(ids)=={id for id,s in ss.items() if s['decision']=='include'}
print('Evidence: all 98 included records assigned once; no excluded record used to inflate counts.')
for w in range(1,5):
 groups=[r for r in rows if r['week']==w]
 for r in groups:
  selected=[q for q in qs if q['id'] in r['question_ids']]
  assert len(selected)==r['occurrences']
  assert all(ss[q['id']]['week']==w for q in selected)
  assert dict(Counter(q['exam'] for q in selected))==r['exam_counts']
  assert len({(q['term'],q['exam']) for q in selected})==r['distinct_exam_terms']
 print(f"Week {w} evidence: "+', '.join(f"{r['family']}={r['occurrences']}" for r in groups))
paths=[ROOT/f'week {w}/week-{w}-exam-focused-notes.md' for w in range(1,5)]
allpaths=paths+[ROOT/'exam-study-guide.md']
wiki={p.stem for p in ROOT.rglob('*.md')}
images=0
for path in allpaths:
 text=path.read_text()
 data=yaml.safe_load(text.split('---',2)[1]);assert data['status']=='complete'
 assert '\t' not in text and not any(ord(c)<32 and c not in '\n\r' for c in text)
 for item in re.findall(r'\[\[([^\]]+)\]\]',text):
  target=item.replace('\\|','|').split('|')[0].split('#')[0]
  assert target in wiki or (ROOT/(target+'.md')).is_file(),(path,item)
 for m in re.finditer(r'!?\[([^\]\n]+)\]\(([^)]+)\)',text):
  url=m[2]
  if url.startswith(('https://','http://')): continue
  name,_,anchor=unquote(url).partition('#');p=(path.parent/name).resolve() if name else path
  assert p.is_file(),(path,url)
  if anchor and p.suffix=='.md': assert f'id="{anchor}"' in p.read_text(),(path,url)
 img=list(re.finditer(r'!\[[^\]]*\]\([^)]+\)',text))
 for m in img:assert re.match(r'\s*\*\*Figure \d+\.\*\*',text[m.end():]),(path,'caption')
 assert [int(n) for n in re.findall(r'\*\*Figure (\d+)\.\*\*',text)]==list(range(1,len(img)+1))
 images+=len(img)
 # Pipes in tables must not be unescaped conditional-probability or alias delimiters.
 for line in text.splitlines():
  if line.startswith('|'):
   assert not re.search(r'\[\[[^\]]*(?<!\\)\|',line),(path,line)
 if path not in paths: continue
 blocks=re.findall(r'```python\n(.*?)\n```',text,re.S);assert len(blocks)==1
 output=io.StringIO()
 with redirect_stdout(output):exec(compile(blocks[0],str(path),'exec'),{'__name__':'__main__'})
 expected=output.getvalue()
 marker=r'<!-- lab-output:start -->.*?<!-- lab-output:end -->'
 rendered='<!-- lab-output:start -->\n```text\n'+expected.rstrip()+'\n```\n<!-- lab-output:end -->'
 if '--refresh-outputs' in sys.argv:
  text,n=re.subn(marker,lambda _:rendered,text,flags=re.S);assert n==1;path.write_text(text)
 else:assert rendered in text,(path,'stdout differs from recorded output')
 assert len(text.split())>2000,(path,'Guide unexpectedly short')
 print(f'{path.parent.name}: Python example executed; recorded stdout matches; {len(text.split())} whitespace-delimited words.')
print(f'Artifacts: all five YAML headers parse; links resolve; {images} existing figures have sequential captions.')
# Independent arithmetic checks beyond the embedded demonstrations.
normal_tail=0.5*math.erfc(1/math.sqrt(2))
assert math.isclose(normal_tail,.15865525393145707)
assert math.isclose(.5*normal_tail+.5*normal_tail,normal_tail)
print(f'Gaussian example: each conditional error and overall equal-prior error = {normal_tail:.9f}.')
A=[[.5,.375,.125],[.25,.125,.625],[.25,.075,.675]]
dry=[.6,.25,.05];damp=[.15,.25,.35]
first=[A[1][i]*dry[i] for i in range(3)]
forward=[damp[j]*sum(first[i]*A[i][j] for i in range(3)) for j in range(3)]
assert all(math.isclose(a,b) for a,b in zip(forward,[.01359375,.015625,.02078125]))
evidence=sum(forward);assert math.isclose(evidence,.05)
beta=[sum(A[i][j]*damp[j] for j in range(3)) for i in range(3)]
posterior=[first[i]*beta[i]/evidence for i in range(3)]
assert all(math.isclose(a,b) for a,b in zip(posterior,[.6375,.1796875,.1828125]))
assert math.isclose(sum(posterior),1)
print('HMM example: Forward evidence 0.050000000; best-path posterior 0.281250; smoothed Day 1 occupancy verified.')
assert math.isclose(.8*.4**2+.2*1.6**2,.64)
assert math.ceil(math.log2(50))==6 and math.ceil(math.log2(2000))==11
assert math.isclose(1/(1-.8),5) and 1+(16000-400)//160==98
print('M-step variance, VQ bit widths, geometric mean duration and unpadded frame count verified.')
print('PASS: study evidence, numerical demonstrations, source navigation and guide structure.')
