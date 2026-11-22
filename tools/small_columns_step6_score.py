"""Score the fresh browser runs with the frozen original one-to-one matcher."""
import json, shutil, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'output/small_columns_step6'
for mode in ('off','on'):
 d=out/('score_'+mode);d.mkdir(exist_ok=False)
 for name in ('正文方剂表_步骤1标注.json','正文方剂表_步骤6人工转写结果.json'):
  shutil.copyfile(root/'docs'/name,d/name)
 shutil.copyfile(out/('browser_'+mode)/'browser_results.json',d/'正文方剂表_步骤6网页原始结果.json')
 subprocess.run([sys.executable,str(root/'tools/score_formula_stage6.py'),'--input-dir',str(d),'--output',str(d/'score.json')],cwd=root,check=True)
 x=json.loads((d/'score.json').read_text(encoding='utf8'))
 # Preserve the original strict three-field provenance score; the historical scorer
 # also emits a looser name-source-only metric, which is not used for this gate.
 print(mode,'strict',json.dumps(x['ocr_total'],ensure_ascii=False),'clicks',json.dumps(x['browser_source_checks'],ensure_ascii=False),flush=True)
