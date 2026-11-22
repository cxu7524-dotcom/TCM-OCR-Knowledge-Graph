"""Compare both printed tables' entire cell grids to the previous frozen run."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
before=json.loads((ROOT/'output/formula_stage6/run-v6az4xah/正文方剂表_步骤6网页原始结果.json').read_text(encoding='utf8'))
after=json.loads((ROOT/'docs/正文方剂表_步骤6网页原始结果.json').read_text(encoding='utf8'))
out={}
for key in ('p0011','p0015'):
    a=before['printed'][key];b=after['printed'][key]
    at=[t for t in a['api']['tables'] if t.get('source_type')=='原书表格']
    bt=[t for t in b['api']['tables'] if t.get('source_type')=='原书表格']
    arows=[t['rows'] for t in at]; brows=[t['rows'] for t in bt]
    same=arows==brows and len(brows)==1 and len(brows[0])==6 and all(len(r)==3 for r in brows[0])
    out[key]={'full_grid_equal':same,'table_count':len(bt),'rows':len(brows[0]) if brows else 0,
              'columns':[len(r) for r in brows[0]] if brows else [],
              'cell_count':sum(len(r) for r in brows[0]) if brows else 0,
              'target_click_pass':b.get('pass',False),'observed_rows':brows}
    assert same and b.get('pass',False),key
dest=Path(after['run'])/'printed_full_grid_regression.json'
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False))
