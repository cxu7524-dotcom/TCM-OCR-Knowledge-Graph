"""步骤6固定五页真实浏览器上传和逐字段点击；失败保留，不修改模型或规则。"""
import json,socket,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright,TimeoutError as PlaywrightTimeout
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'output'/'formula_stage6';out.mkdir(parents=True,exist_ok=True)
run=Path(tempfile.mkdtemp(prefix='run-',dir=out));log=open(run/'server.log','w',encoding='utf8')
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
server=subprocess.Popen([sys.executable,str(ROOT/'tools/formula_stage4_test_server.py'),str(run),str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));base=f'http://127.0.0.1:{port}'
records={'run':str(run),'pages':{},'printed':{},'errors':[]}
def ready():
 for _ in range(160):
  if server.poll() is not None:raise RuntimeError('server exited; inspect '+str(run/'server.log'))
  try:
   with urllib.request.urlopen(base+'/api/meta',timeout=2):return
  except Exception:time.sleep(.25)
 raise RuntimeError('server startup timeout')
def selected(pg):
 return pg.evaluate('''() => ({blocks:[...document.querySelectorAll('.bbox.sel')].map(e=>{let b=S.ocr.blocks[+e.dataset.i];return {line:b.line_id,ref:b.block_ref,text:b.text,bbox:b.bbox}}),detail:document.getElementById('tableDetail').innerText})''')
def upload(pg,path):
 pg.click('.step-btn[data-step="1"]');pg.set_input_files('#fileInput',str(path));pg.click('#btnUpload')
 pg.wait_for_function("() => document.getElementById('s1hint').textContent.includes('已加载页面')",timeout=240000)
 pid=pg.evaluate('() => S.pageId');pg.click('.step-btn[data-step="2"]')
 pg.wait_for_function('(pid) => S.ocr && S.ocr.page_id === pid',arg=pid,timeout=60000)
 return pid
try:
 ready()
 with sync_playwright() as p:
  browser=p.chromium.launch(channel='chrome',headless=True);ctx=browser.new_context(viewport={'width':1560,'height':1100});pg=ctx.new_page()
  pg.on('pageerror',lambda e:records['errors'].append('pageerror: '+str(e)))
  pg.on('console',lambda m:records['errors'].append('console: '+m.text) if m.type=='error' else None)
  pg.goto(base,wait_until='networkidle')
  for n in (37,39,40,41,50):
   key=str(n);entry={'page':n,'upload':'未完成','clicks':[]};records['pages'][key]=entry
   try:
    img=ROOT/'古籍评测集'/'医家类'/'思濟堂方書'/f'page_{n}.png';pid=upload(pg,img);entry['page_id']=pid;entry['upload']='成功'
    api=ctx.request.get(base+f'/api/pages/{pid}/ocr');entry['api_http']=api.status;entry['api']=api.json()
    entry['image_visible']=pg.locator('#pageCanvas img.base').count()>0
    pg.locator('#tableCard').screenshot(path=str(run/f'page_{n}_tables.png'))
    # 点查每张正文表的方名和每一行药材、剂量；包括失败/无法定位。
    panels=pg.locator('#tableWrap .formula-panel');entry['rendered_panels']=panels.count()
    for k in range(panels.count()):
     panel=panels.nth(k);typ=panel.locator('.formula-heading .badge').first.inner_text()
     if typ!='正文抽取':continue
     title=panel.locator('button.formula-name')
     if title.count() and title.is_enabled():
      try:title.click(timeout=10000);entry['clicks'].append({'table':k,'field':'方名','value':title.inner_text(),**selected(pg)})
      except Exception as exc:entry['clicks'].append({'table':k,'field':'方名','error':str(exc)})
     for td in panel.locator('td.cell').all():
      try:
       val=td.locator('.ctext').inner_text(); rr=td.get_attribute('data-r');cc=td.get_attribute('data-c')
       td.click(timeout=10000);entry['clicks'].append({'table':k,'row':int(rr),'column':int(cc),'field':'药材' if cc=='1' else '剂量','value':val,**selected(pg)})
      except Exception as exc:entry['clicks'].append({'table':k,'field':'药材/剂量','error':str(exc)})
    hints=pg.locator('.continuation-hint');entry['visible_hints']=hints.count()
    if hints.count():
     try:hints.first.click(timeout=10000);entry['hint_click']=selected(pg)
     except Exception as exc:entry['hint_click']={'error':str(exc)}
    pg.locator('#pageCanvas').screenshot(path=str(run/f'page_{n}_last_highlight.png'))
    print('page',n,'success','tables',[(t.get('source_type'),len(t.get('rows',[]))) for t in entry['api'].get('tables',[])],'clicks',len(entry['clicks']),flush=True)
   except Exception as exc:
    entry['error']=repr(exc);records['errors'].append(f'page {n}: {exc}');print('page',n,'FAILED',repr(exc),flush=True)
    try:pg.screenshot(path=str(run/f'page_{n}_failure.png'))
    except Exception:pass
   (run/f'page_{n}_record.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf8')
  for tag in ('p0011','p0015'):
   entry={'upload':'未完成'};records['printed'][tag]=entry
   try:
    img=ROOT/'data'/'samples'/'synthetic'/'pages'/f'SYN-2026-001__{tag}.png';pid=upload(pg,img);entry['page_id']=pid;entry['upload']='成功';entry['api']=ctx.request.get(base+f'/api/pages/{pid}/ocr').json()
    cell=pg.locator('#tableWrap .formula-panel').first.locator('td.cell[data-r="1"][data-c="1"]');entry['value']=cell.locator('.ctext').inner_text();cell.click(timeout=10000);entry['click']=selected(pg)
    entry['pass']=(entry['value']=='三兩' and len(entry['click']['blocks'])==1 and entry['click']['blocks'][0]['line']=='R006' and entry['click']['blocks'][0]['bbox']==[342,323,405,359])
    pg.locator('#pageCanvas').screenshot(path=str(run/f'{tag}_dose.png'));print(tag,'pass=',entry['pass'],flush=True)
   except Exception as exc:entry['error']=repr(exc);entry['pass']=False;records['errors'].append(f'{tag}: {exc}');print(tag,'FAILED',repr(exc),flush=True)
  records['browser_version']=browser.version;ctx.close();browser.close()
 (ROOT/'docs'/'正文方剂表_步骤6网页原始结果.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 print('saved',run,flush=True)
finally:
 server.terminate()
 try:server.wait(timeout=15)
 except subprocess.TimeoutExpired:server.kill();server.wait()
 log.close()
