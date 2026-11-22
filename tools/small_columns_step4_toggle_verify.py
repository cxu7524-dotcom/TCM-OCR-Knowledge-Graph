"""Verify same-page enabled -> disabled upload with a shared isolated store."""
import json,os,socket,subprocess,sys,time,urllib.error,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step4';run=O/'http_toggle';assert not run.exists();run.mkdir()
image=ROOT/'古籍评测集/医家类/思濟堂方書/page_35.png';pid='S4TOGGLE35';result={}
def call(base,path,data=None,headers=None):
 request=urllib.request.Request(base+path,data=data,headers=headers or {},method='POST' if data is not None else 'GET')
 with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
for flag in ('1','0'):
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 env=os.environ.copy();env['TCM_OCR_SMALL_COLUMNS']=flag;log=(run/('on.log' if flag=='1' else 'off.log')).open('w',encoding='utf8')
 proc=subprocess.Popen([sys.executable,str(ROOT/'tools/formula_stage4_test_server.py'),str(run),str(port)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));base=f'http://127.0.0.1:{port}'
 try:
  for _ in range(120):
   if proc.poll() is not None:raise RuntimeError('server exited')
   try:call(base,'/api/meta');break
   except (OSError,urllib.error.URLError):time.sleep(.25)
  boundary=uuid.uuid4().hex;data=(f'--{boundary}\r\nContent-Disposition: form-data; name="page_id"\r\n\r\n{pid}\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\nContent-Type: image/png\r\n\r\n').encode()+image.read_bytes()+b'\r\n'+f'--{boundary}--\r\n'.encode();call(base,'/api/pages',data,{'Content-Type':f'multipart/form-data; boundary={boundary}'});result[flag]=call(base,f'/api/pages/{pid}/ocr')
 finally:
  proc.terminate()
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close()
on=result['1'];off=result['0'];evidence=lambda r:[b for b in r['blocks'] if b['kind']=='small_column_evidence'];assert evidence(on) and not evidence(off)
assert any(r[1:]==['白术','五錢'] for t in on['tables'] for r in t['rows']);assert not any(r[1:]==['白术','五錢'] for t in off['tables'] for r in t['rows'])
normal=lambda r:sorted((b['bbox'],b['text']) for b in r['blocks'] if b['kind']!='small_column_evidence');assert normal(on)==normal(off)
(O/'toggle_verification.json').write_text(json.dumps({'same_page_id':pid,'enabled_evidence':len(evidence(on)),'disabled_evidence':len(evidence(off)),'original_blocks_unchanged':True,'enabled_and_disabled_results':result},ensure_ascii=False,indent=2),encoding='utf8');print('enabled evidence',len(evidence(on)),'disabled evidence',len(evidence(off)),'original blocks unchanged',flush=True)
