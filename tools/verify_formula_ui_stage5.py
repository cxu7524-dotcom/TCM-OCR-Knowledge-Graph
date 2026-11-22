"""步骤 5 浏览器实点：真实开发页上传、受控跨块及两张合成规整表。"""
import json,socket,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'output'/'formula_stage5';out.mkdir(parents=True,exist_ok=True)
run=Path(tempfile.mkdtemp(prefix='run-',dir=out));log=open(run/'server.log','w',encoding='utf8')
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
server=subprocess.Popen([sys.executable,str(ROOT/'tools/formula_stage5_test_server.py'),str(run),str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));base=f'http://127.0.0.1:{port}'
records={'run':str(run),'real':{},'controlled':{},'printed':{},'errors':[]}
def ready():
 for _ in range(160):
  if server.poll() is not None:raise RuntimeError('server exited: '+str(run/'server.log'))
  try:
   with urllib.request.urlopen(base+'/api/meta',timeout=2):return
  except Exception:time.sleep(.25)
 raise RuntimeError('server startup timeout')
def selected(pg):
 return pg.evaluate('''() => ({blocks:[...document.querySelectorAll('.bbox.sel')].map(e=>{let b=S.ocr.blocks[+e.dataset.i];return {line:b.line_id,ref:b.block_ref,text:b.text,bbox:b.bbox}}),detail:document.getElementById('tableDetail').innerText,block_detail:document.getElementById('blockDetail').innerText})''')
def upload(pg,path):
 pg.click('.step-btn[data-step="1"]');pg.set_input_files('#fileInput',str(path));pg.click('#btnUpload')
 pg.wait_for_function("() => document.getElementById('s1hint').textContent.includes('已加载页面')",timeout=240000)
 pid=pg.evaluate('() => S.pageId');pg.click('.step-btn[data-step="2"]')
 pg.wait_for_function("(pid) => S.ocr && S.ocr.page_id===pid && document.querySelector('#tableWrap table.fangji')",arg=pid,timeout=60000)
 return pid
def click_record(pg,locator):
 locator.click();return selected(pg)
try:
 ready()
 with sync_playwright() as p:
  browser=p.chromium.launch(channel='chrome',headless=True);ctx=browser.new_context(viewport={'width':1560,'height':1100});pg=ctx.new_page()
  pg.on('pageerror',lambda e:records['errors'].append(str(e)))
  pg.on('console',lambda m:records['errors'].append('console: '+m.text) if m.type=='error' else None)
  pg.goto(base,wait_until='networkidle');pg.wait_for_selector('#pageList .page-item')
  path=ROOT/'古籍评测集'/'医家类'/'思濟堂方書'/'page_35.png';pid=upload(pg,path)
  pg.wait_for_selector('#pageCanvas img.base');pg.wait_for_function("() => document.querySelector('#pageCanvas img.base').complete && document.querySelector('#pageCanvas img.base').naturalWidth>0")
  panels=pg.locator('#tableWrap .formula-panel');names=panels.locator('button.formula-name').all_text_contents();assert names[:2]==['三化湯','六君子湯'],names
  assert panels.count()==2
  records['real']['page_id']=pid;records['real']['names']=names
  records['real']['first_name']=click_record(pg,panels.nth(0).locator('button.formula-name'))
  assert records['real']['first_name']['blocks'][0]['text']=='三化湯'
  records['real']['first_herb']=click_record(pg,panels.nth(0).locator('td.cell[data-r="1"][data-c="1"]'))
  assert '羌活' in records['real']['first_herb']['blocks'][0]['text']
  records['real']['uncertain_dose']=click_record(pg,panels.nth(0).locator('td.cell[data-r="1"][data-c="2"]'))
  assert len(records['real']['uncertain_dose']['blocks'])==2 and '待核' in records['real']['uncertain_dose']['detail'] and 'OCR 原文' in records['real']['uncertain_dose']['detail']
  pg.locator('#tableCard').screenshot(path=str(run/'real_tables.png'));pg.locator('#pageCanvas').screenshot(path=str(run/'real_pending_highlight.png'))
  records['real']['second_name']=click_record(pg,panels.nth(1).locator('button.formula-name'))
  assert records['real']['second_name']['blocks'][0]['text']=='六君子湯'
  records['real']['second_herb']=click_record(pg,panels.nth(1).locator('td.cell[data-r="1"][data-c="1"]'))
  records['real']['second_dose']=click_record(pg,panels.nth(1).locator('td.cell[data-r="1"][data-c="2"]'))
  assert records['real']['second_herb']['blocks'] and records['real']['second_dose']['blocks']
  # 切页后重新打开原页，再实点同一待核格。
  pg.click('.step-btn[data-step="1"]');fixture='P20261119000000-abc123';pg.wait_for_selector(f'.page-item[data-pid="{fixture}"]');pg.locator(f'.page-item[data-pid="{fixture}"]').click();pg.wait_for_function('(id) => S.pageId===id',arg=fixture)
  pg.locator(f'.page-item[data-pid="{pid}"]').click();pg.wait_for_function('(id) => S.pageId===id',arg=pid);pg.click('.step-btn[data-step="2"]')
  pg.wait_for_function('(id) => S.ocr && S.ocr.page_id===id && document.querySelectorAll("#tableWrap .formula-panel").length===2',arg=pid)
  again=click_record(pg,pg.locator('#tableWrap .formula-panel').nth(0).locator('td.cell[data-r="1"][data-c="2"]'))
  assert [b['ref'] for b in again['blocks']]==[b['ref'] for b in records['real']['uncertain_dose']['blocks']]
  records['real']['reopened_uncertain_dose']=again
  # 浏览器整页刷新后从历史列表重开，再核对存储恢复和定位。
  pg.reload(wait_until='networkidle');pg.wait_for_selector(f'.page-item[data-pid="{pid}"]',timeout=30000)
  pg.locator(f'.page-item[data-pid="{pid}"]').click();pg.wait_for_function('(id) => S.pageId===id',arg=pid)
  pg.click('.step-btn[data-step="2"]');pg.wait_for_function('(id) => S.ocr && S.ocr.page_id===id && document.querySelectorAll("#tableWrap .formula-panel").length===2',arg=pid)
  restored=click_record(pg,pg.locator('#tableWrap .formula-panel').nth(0).locator('td.cell[data-r="1"][data-c="2"]'))
  assert [b['ref'] for b in restored['blocks']]==[b['ref'] for b in again['blocks']]
  records['real']['browser_reloaded_uncertain_dose']=restored
  # 受控输入：合成影像与预置 OCR，演示共享剂量跨两个块、缺载和续页。
  pg.click('.step-btn[data-step="1"]');pg.wait_for_selector(f'.page-item[data-pid="{fixture}"]');pg.locator(f'.page-item[data-pid="{fixture}"]').click();pg.click('.step-btn[data-step="2"]')
  pg.wait_for_function('(id) => S.ocr && S.ocr.page_id===id && document.querySelectorAll("#tableWrap .formula-panel").length>=3',arg=fixture)
  panels=pg.locator('#tableWrap .formula-panel');records['controlled']['page_id']=fixture;records['controlled']['names']=panels.locator('button.formula-name').all_text_contents()
  first=click_record(pg,panels.nth(0).locator('td.cell[data-r="1"][data-c="2"]')); second=click_record(pg,panels.nth(0).locator('td.cell[data-r="2"][data-c="2"]'))
  assert [b['line'] for b in first['blocks']]==['R003','R004'] and [b['line'] for b in second['blocks']]==['R003','R004']
  records['controlled']['shared_first']=first;records['controlled']['shared_second']=second
  keycell=panels.nth(0).locator('td.cell[data-r="1"][data-c="2"]');keycell.focus();pg.keyboard.press('Enter')
  keyed=selected(pg);assert [b['line'] for b in keyed['blocks']]==['R003','R004']
  records['controlled']['keyboard_shared']=keyed
  pg.locator('#pageCanvas').screenshot(path=str(run/'controlled_shared.png'))
  absent=click_record(pg,panels.nth(1).locator('td.cell[data-r="1"][data-c="2"]'))
  assert '原文未载' in absent['detail'] and [b['line'] for b in absent['blocks']]==['R007']
  records['controlled']['absent']=absent
  assert '可能续页' in panels.nth(2).inner_text();records['controlled']['continuation_visible']=True
  pg.locator('#tableCard').screenshot(path=str(run/'controlled_tables.png'))
  for tag in ('p0011','p0015'):
   path=ROOT/'data'/'samples'/'synthetic'/'pages'/f'SYN-2026-001__{tag}.png';q=upload(pg,path)
   cell=pg.locator('#tableWrap .formula-panel').first.locator('td.cell[data-r="1"][data-c="1"]')
   assert cell.locator('.ctext').inner_text()=='三兩'
   hit=click_record(pg,cell);assert len(hit['blocks'])==1 and hit['blocks'][0]['line']=='R006' and hit['blocks'][0]['bbox']==[342,323,405,359]
   records['printed'][tag]={'page_id':q,'click':hit};pg.locator('#pageCanvas').screenshot(path=str(run/(tag+'_dose.png')))
  assert not records['errors'],records['errors']
  records['browser_version']=browser.version;ctx.close();browser.close()
 (ROOT/'docs'/'正文方剂表_步骤5浏览器验证.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'run':str(run),'real_page':records['real']['page_id'],'real_names':records['real']['names'],'controlled_names':records['controlled']['names'],'printed':list(records['printed']),'errors':records['errors']},ensure_ascii=False),flush=True)
finally:
 server.terminate()
 try:server.wait(timeout=15)
 except subprocess.TimeoutExpired:server.kill();server.wait()
 log.close()
