"""Freeze an unexamined negative page, then run the known regressions via HTTP."""
import hashlib, json, os, socket, subprocess, sys, time, urllib.request, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'small_columns_step6'
OUT.mkdir(parents=True, exist_ok=True)
imgs = {
 'p35': ROOT/'古籍评测集/医家类/思濟堂方書/page_35.png',
 'wenyi12_known': ROOT/'古籍评测集/医家类/温疫方論/page_12.png',
 'wenyi08_negative': ROOT/'古籍评测集/医家类/温疫方論/page_8.png',
 'p34': ROOT/'古籍评测集/医家类/思濟堂方書/page_34.png',
}
manifest = {'holdout': {'image': str(imgs['wenyi08_negative']),
 'sha256': hashlib.sha256(imgs['wenyi08_negative'].read_bytes()).hexdigest(),
 'annotation': '整页为疫气、物类及药物制气的论述，未见具备方名和逐味组成的方剂；应出正文方剂行 0。仅作为负页检查，不计算召回率。',
 'gt_formula_rows': 0, 'reviewed_before_inference': True},
 'regressions': {k: {'image': str(v), 'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in imgs.items() if k != 'wenyi08_negative'}}
mf = OUT/'extra_manifest.json'
assert not mf.exists()
mf.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
def post(base, path, pid):
 boundary='test-'+uuid.uuid4().hex
 data=(f'--{boundary}\r\nContent-Disposition: form-data; name="page_id"\r\n\r\n{pid}\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\nContent-Type: image/png\r\n\r\n').encode()+path.read_bytes()+b'\r\n'+f'--{boundary}--\r\n'.encode()
 req=urllib.request.Request(base+'/api/pages',data=data,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'},method='POST')
 with urllib.request.urlopen(req,timeout=300) as r:return json.load(r)
def get(base,path):
 with urllib.request.urlopen(base+path,timeout=180) as r:return json.load(r)
for mode in ('off','on'):
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 run=OUT/('http_'+mode);run.mkdir(exist_ok=False)
 env=os.environ.copy();env['TCM_OCR_SMALL_COLUMNS']='1' if mode=='on' else '0';env['TCM_OCR_SMALL_COLUMNS_SERVER']=env['TCM_OCR_SMALL_COLUMNS']
 with (run/'server.log').open('w',encoding='utf8') as log:
  proc=subprocess.Popen([sys.executable,str(ROOT/'tools/formula_stage4_test_server.py'),str(run),str(port)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));base=f'http://127.0.0.1:{port}'
  try:
   for _ in range(160):
    if proc.poll() is not None:raise RuntimeError('server exited')
    try:get(base,'/api/meta');break
    except OSError:time.sleep(.25)
   else:raise RuntimeError('server timeout')
   for name,path in imgs.items():
    pid='S6'+mode.upper()+name.upper().replace('_','');up=post(base,path,pid);read=get(base,f'/api/pages/{pid}/ocr')
    assert up['model_loaded'] and not up['is_demo'] and up['tables']==read['tables']
    (run/(name+'.json')).write_text(json.dumps({'upload':up,'read':read},ensure_ascii=False,indent=2),encoding='utf8')
    print(mode,name,'blocks',len(read['blocks']),'tables',len(read['tables']),flush=True)
  finally:
   proc.terminate()
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
