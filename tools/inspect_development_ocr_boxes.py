import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'output/formula_stage6/开发页竖排路径对照.json'
data=json.loads(path.read_text(encoding='utf8'))
for n,page in data['pages'].items():
 print('\nPAGE',n)
 for mode in ('as_installed',):
  for i,b in enumerate(page[mode]['blocks']):
   x1,y1,x2,y2=b['bbox']
   if 180<=y1<=700:
    print(i,b['bbox'],round(b['confidence'],2),b['text'])
