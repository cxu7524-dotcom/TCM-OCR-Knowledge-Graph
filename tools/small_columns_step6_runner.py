"""Run the existing browser acceptance in a new, isolated A/B evidence directory."""
import hashlib, json, os, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'output' / 'small_columns_step6'
out.mkdir(parents=True, exist_ok=True)
annotation = root / 'docs' / '正文方剂表_步骤1标注.json'
sha = hashlib.sha256(annotation.read_bytes()).hexdigest()
assert sha == 'a8ded6455d01a64a163f309efd16d06d7204ada55eca8b772a4becf9732025b9'
source = (root / 'tools' / 'verify_formula_browser_stage6.py').read_text(encoding='utf-8')
source = source.replace("out=ROOT/'output'/'formula_stage6'", "out=ROOT/'output'/'small_columns_step6'/('browser_'+os.environ['STEP6_MODE'])")
source = source.replace('import json,socket,subprocess,sys,tempfile,time,urllib.request', 'import json,socket,subprocess,sys,tempfile,time,urllib.request,os')
source = source.replace("(ROOT/'docs'/'正文方剂表_步骤6网页原始结果.json').write_text", "(out/'browser_results.json').write_text")
generated = out / 'browser_runner.py'
generated.write_text(source, encoding='utf-8')
# The copied runner assumes its own location is tools/; preserve the repository root.
source = source.replace("ROOT=Path(__file__).resolve().parents[1]", "ROOT=Path(__file__).resolve().parents[2]")
generated.write_text(source, encoding='utf-8')
for mode in ('off', 'on'):
    env = os.environ.copy()
    env['STEP6_MODE'] = mode
    env['TCM_OCR_SMALL_COLUMNS'] = '1' if mode == 'on' else '0'
    env['TCM_OCR_SMALL_COLUMNS_SERVER'] = '1' if mode == 'on' else '0'
    print('starting', mode, flush=True)
    completed = subprocess.run([sys.executable, str(generated)], cwd=root, env=env)
    print('finished', mode, completed.returncode, flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
