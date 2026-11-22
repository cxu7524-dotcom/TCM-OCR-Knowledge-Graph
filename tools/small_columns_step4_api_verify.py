"""Real local HTTP upload/GET with isolated stores and explicit A/B flag."""
import json,os,socket,subprocess,sys,time,urllib.error,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'output/small_columns_step4';O.mkdir(exist_ok=True)
target=O/'api_ab_results.json';assert not target.exists()
images={'p35':ROOT/'古籍评测集/医家类/思濟堂方書/page_35.png','newbook':ROOT/'古籍评测集/医家类/温疫方論/page_12.png','p0011':ROOT/'data/samples/synthetic/pages/SYN-2026-001__p0011.png','p0015':ROOT/'data/samples/synthetic/pages/SYN-2026-001__p0015.png'}
def post(base,path,pid):
 boundary='test-'+uuid.uuid4().hex
 data=(f'--{boundary}\r\nContent-Disposition: form-data; name="page_id"\r\n\r\n{pid}\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\nContent-Type: image/png\r\n\r\n').encode()+path.read_bytes()+b'\r\n'+f'--{boundary}--\r\n'.encode()
 req=urllib.request.Request(base+'/api/pages',data=data,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'},method='POST')
 with urllib.request.urlopen(req,timeout=240) as r:return json.load(r)
def get(base,path):
 with urllib.request.urlopen(base+path,timeout=180) as r:return json.load(r)
results={}
for flag in ('0','1'):
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 run=O/('http_on' if flag=='1' else 'http_off');assert not run.exists();run.mkdir();log=(run/'server.log').open('w',encoding='utf8');env=os.environ.copy();env['TCM_OCR_SMALL_COLUMNS']=flag
 proc=subprocess.Popen([sys.executable,str(ROOT/'tools/formula_stage4_test_server.py'),str(run),str(port)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));base=f'http://127.0.0.1:{port}'
 try:
  for _ in range(120):
   if proc.poll() is not None:raise RuntimeError('server exited; see '+str(run/'server.log'))
   try:get(base,'/api/meta');break
   except (OSError,urllib.error.URLError):time.sleep(.25)
  else:raise RuntimeError('server not ready')
  for name,path in images.items():
   pid=('S4ON' if flag=='1' else 'S4OFF')+name.upper();up=post(base,path,pid);read=get(base,f'/api/pages/{pid}/ocr');assert up['model_loaded'] and not up['is_demo'];assert up['tables']==read['tables'];results[flag+'_'+name]={'upload':up,'read':read};print(flag,name,'blocks',len(read['blocks']),'tables',len(read['tables']),flush=True)
 finally:
  proc.terminate()
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close()
target.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
for mode in ('0','1'):
 data=results[mode+'_p35']['read']
 for table in data['tables']:
  if table['source_type']=='正文抽取':
   for i,row in enumerate(table['rows']):
    if any('白术' in str(x) or '白朮' in str(x) for x in row):print('P35',mode,ascii(row),ascii(table['field_details'][i]),flush=True)
print('saved',target,flush=True)
