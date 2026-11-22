"""步骤 4：隔离真实 HTTP 上传、重读、幂等与混合图像接口记录。"""
import json, socket, subprocess, sys, tempfile, time, urllib.request, urllib.error, uuid
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'output'/'formula_stage4';out.mkdir(parents=True,exist_ok=True)
run=Path(tempfile.mkdtemp(prefix='run-',dir=out))
with socket.socket() as sock:
 sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
log=open(run/'server.log','w',encoding='utf8')
server=subprocess.Popen([sys.executable,str(ROOT/'tools/formula_stage4_test_server.py'),str(run),str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
base=f'http://127.0.0.1:{port}'
def get(url):
 with urllib.request.urlopen(base+url,timeout=180) as r:return json.load(r)
def post(path,pid):
 boundary='test-'+uuid.uuid4().hex
 def field(name,value):return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode('utf8')
 content=field('page_id',pid)
 content+=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\nContent-Type: image/png\r\n\r\n').encode('utf8')+path.read_bytes()+b'\r\n'+f'--{boundary}--\r\n'.encode('ascii')
 req=urllib.request.Request(base+'/api/pages',data=content,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'},method='POST')
 with urllib.request.urlopen(req,timeout=240) as r:return json.load(r)
def summary(name,up,read):
 assert up['tables']==read['tables']
 assert up['formula_continuation_hints']==read['formula_continuation_hints']
 refs={b['block_ref'] for b in read['blocks']}
 for table in read['tables']:
  if table['source_type']=='正文抽取':
   for row in table['cell_refs_multi']:
    for cell in row:
     assert all(ref in refs for ref in cell)
 print(name,'engine=',up['engine'],'tables=',[(t['source_type'],t['stats']['rows'],t['stats']['cols']) for t in read['tables']],flush=True)
 return {'upload':up,'read':read}
try:
 for _ in range(120):
  if server.poll() is not None:raise RuntimeError('server exited; see '+str(run/'server.log'))
  try:get('/api/meta');break
  except (OSError,urllib.error.URLError):time.sleep(.25)
 else:raise RuntimeError('server not ready')
 real=ROOT/'古籍评测集'/'医家类'/'思濟堂方書'/'page_35.png'
 table=ROOT/'data'/'samples'/'synthetic'/'pages'/'SYN-2026-001__p0011.png'
 u1=post(real,'S4REAL35'); r1=get('/api/pages/S4REAL35/ocr'); results={'real_35':summary('real_35',u1,r1)}
 assert any(t['source_type']=='正文抽取' for t in r1['tables'])
 t=next(t for t in r1['tables'] if t['source_type']=='正文抽取')
 assert t['cell_statuses'][1][2]=='待核' and t['cell_ref_scopes'][1][2]=='配伍上下文'
 assert t['cell_refs_multi'][1][2] and t['field_details'][1][2]['evidence_context']['sources']
 u2=post(table,'S4TABLE11');r2=get('/api/pages/S4TABLE11/ocr');results['table_11']=summary('table_11',u2,r2)
 assert any(t['source_type']=='原书表格' and t['stats']['rows']==6 and t['stats']['cols']==3 for t in r2['tables'])
 p11=next(t for t in r2['tables'] if t['source_type']=='原书表格')
 expected=[['藥名','劑量','炮製'],['桂枝','三兩','去皮'],['芍藥','三兩','切'],['甘草','二兩','炙'],['生薑','三兩','切'],['大棗','十二枚','擘']]
 assert p11['rows']==expected and all(ref for row in p11['cell_refs'] for ref in row)
 # 真实书中规整网格，内容是脉图而非药量表，单独记录识别效果。
 real_grid=ROOT/'古籍评测集'/'医家类'/'合併脉訣難經太素評林'/'page_7.png'
 ug=post(real_grid,'S4REALGRID7');rg=get('/api/pages/S4REALGRID7/ocr')
 results['real_grid_7']=summary('real_grid_7',ug,rg)
 # 相同页 ID、相同影像走两次完整上传：结构与标识须一致。
 u3=post(real,'S4REAL35');r3=get('/api/pages/S4REAL35/ocr')
 assert u1['tables']==u3['tables']==r3['tables']
 assert u1['formula_continuation_hints']==u3['formula_continuation_hints']==r3['formula_continuation_hints']
 results['repeat']={'equal':True,'table_ids':[t['table_id'] for t in u3['tables']]}
 print('repeat equal',results['repeat']['table_ids'],flush=True)
 # 混合图像由公开扫描与合成规整表拼接，仅用于测试同页双类型接口。
 im1=Image.open(real).convert('RGB');im2=Image.open(table).convert('RGB')
 canvas=Image.new('RGB',(max(im1.width,im2.width),im1.height+im2.height),'white')
 canvas.paste(im1,(0,0));canvas.paste(im2,(0,im1.height));mix=run/'mixed.png';canvas.save(mix)
 um=post(mix,'S4MIXED');rm=get('/api/pages/S4MIXED/ocr');results['mixed']=summary('mixed',um,rm)
 results['mixed']['image_kind']='真实开发扫描与合成表页拼接，仅工程验证'
 assert {t['source_type'] for t in rm['tables']}=={'原书表格','正文抽取'}
 # 从 SQLite 恢复：临时移动 JSON 镜像，不删除，复核结果一致再恢复。
 snap=run/'store'/'pages'/'S4REAL35.json';held=run/'S4REAL35.snapshot.json';snap.rename(held)
 try:
  restored=get('/api/pages/S4REAL35/ocr');assert restored['tables']==r3['tables']
  assert restored['formula_continuation_hints']==r3['formula_continuation_hints']
  results['sqlite_restore']={'equal_tables':True,'equal_hints':True}
 finally:held.rename(snap)
 print('sqlite restore equal',flush=True)
 (ROOT/'docs'/'正文方剂表_步骤4接口验证.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 print('saved evidence',flush=True)
finally:
 server.terminate()
 try:server.wait(timeout=15)
 except subprocess.TimeoutExpired:server.kill();server.wait()
 log.close()
