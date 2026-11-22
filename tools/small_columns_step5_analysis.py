import hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ocr.normalize import normalize_text
O=ROOT/'output/small_columns_step5';target=O/'analysis.json';assert not target.exists()
raw=json.loads((O/'recognition_raw.json').read_text(encoding='utf8'));mobile=json.loads((ROOT/'output/small_columns_step4/holdout_ab_v3.json').read_text(encoding='utf8'));server=json.loads((O/'holdout_server_ab_v3.json').read_text(encoding='utf8'));geometry=json.loads((ROOT/'output/small_columns_step3_detector/holdout_fifth_adjudicated.json').read_text(encoding='utf8'));audit={x['id']:x for x in geometry['samples']};
causes={
'P37-01':'自动框通过第五轮目视；两个模型均错读末字半，且主字置信不足，原方表未见对应药行。',
'P39-02':'server 恢复各二錢；这是人參与川附子的共享剂量，现有规则禁止自动分配，目标药行未在当前方表中。',
'P39-03':'mobile 和 server 均读对各三錢；共享归属涉及白茯苓与撿白芍，当前方表未见目标行。',
'P39-04':'两模型的 2x 剂量片段都读对；mobile 两尺度不稳定、主字一尺度置信低于门槛，当前方表未见目标行。',
'P43-05':'两模型均读对五錢且主字可读；当前页未形成可绑定的现有方表，不能新造行。',
'P46-07':'两模型均读对各一錢，但涉及半夏、厚朴、蒼朮共享归属；主字置信偏低且无目标方表。',
'P46-08':'第五轮主字框有裁笔；两个模型均未读对八分，server 高置信漏首字，不能确认剂量。',
'P50-09':'主字框末笔裁切，主字被识成黃苓；剂量涉及共享，mobile 两尺度末字不一致，server 把半误为牛。',
'P34-10':'自动框未出现，主字第二字人工仍有疑义；人工参考框可读十枚仅为诊断，不计自动得分。',
'P34-11':'server 自动右轨两尺度恢复一錢二分；当前页没有可绑定的已识别方表。',
'P49-12':'两模型右轨均读对七枚；主字置信不足且当前页无可绑定方表。'}
rows=[]
for a,b in zip(mobile['samples'],server['samples']):
 assert a['id']==b['id'];sid=a['id'];subset=[v for v in raw['rows'] if v['crop']['sample_id']==sid and v['crop']['role']=='right'];auto=next((v for v in subset if v['crop']['mode']=='auto_v5'),None);manual=next(v for v in subset if v['crop']['mode']=='human_reference');want=normalize_text(a['dose_fragment'])
 def read(v,name):
  r=v['readings'][name+'_2x'];return {'text':r['text'],'confidence':r['confidence'],'dose_fragment_hit':bool(want and want in normalize_text(r['text'])),'bbox':v['crop']['bbox']}
 rows.append({'id':sid,'target':a['dose_fragment'],'fifth_geometry_correct':a['fifth_round_visual_box_correct'],'auto_geometry_exists':auto is not None,'auto_mobile_2x':read(auto,'mobile') if auto else None,'auto_server_2x':read(auto,'server') if auto else None,'manual_mobile_2x':read(manual,'mobile'),'manual_server_2x':read(manual,'server'),'step4_mobile_fragment_hit':a['enabled_hit'],'selective_server_fragment_hit':b['enabled_hit'],'strict_formula_row_correct':b['strict_formula_row_correct'],'interpretation':causes[sid]})
timings={}
for model in ('mobile','server'):
 vals=[r['readings'][model+'_2x']['elapsed_ms'] for r in raw['rows'] if r['crop']['mode']=='auto_v5' and r['crop']['sample_id'].startswith('P')];timings[model]={'n':len(vals),'median_ms':round(statistics.median(vals),2),'mean_ms':round(statistics.mean(vals),2)}
new=[]
for r in raw['rows']:
 if r['crop']['sample_id'].startswith('W12-'):
  new.append({'id':r['crop']['sample_id'],'role':r['crop']['role'],'bbox':r['crop']['bbox'],'expected':r['crop']['expected_text'],'mobile':r['readings']['mobile_2x'],'server':r['readings']['server_2x'],'manual_box_only':True})
base_attach=sum(v['audit']['accepted'] for v in mobile['pages'].values());server_attach=sum(v['audit']['accepted'] for v in server['pages'].values());assert base_attach==server_attach==0
data={'scope':'fixed 11 previously seen local samples, identical v5 auto crops; reference crops and new book manual crops reported separately','fixed_total':11,'fixed_auto_geometry_missing':sum(not x['auto_geometry_exists'] for x in rows),'fixed_auto_geometry_failed_manual_review':sum(x['auto_geometry_exists'] and not x['fifth_geometry_correct'] for x in rows),'auto_2x_mobile_hits':sum(bool(x['auto_mobile_2x'] and x['auto_mobile_2x']['dose_fragment_hit']) for x in rows),'auto_2x_server_hits':sum(bool(x['auto_server_2x'] and x['auto_server_2x']['dose_fragment_hit']) for x in rows),'selective_mobile_hits':mobile['enabled_hits'],'selective_server_hits':server['enabled_hits'],'mobile_attached_dose_rows':base_attach,'selective_server_attached_dose_rows':server_attach,'new_wrong_confirmations_in_fixed_pages':0,'limitation':'zero new attachments means zero new confirmed errors in this fixed replay, not a proven error rate on unseen pages','timing':timings,'samples':rows,'newbook_human_reference_only':new,'raw_sha256':hashlib.sha256((O/'recognition_raw.json').read_bytes()).hexdigest()}
target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in data.items() if k not in ('samples','newbook_human_reference_only')},ensure_ascii=True,indent=2))
