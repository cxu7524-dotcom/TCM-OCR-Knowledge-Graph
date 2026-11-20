"""Opt-in, conservative attachment of frozen-v5 crop evidence to existing rows.

Original blocks are not altered. A crop may be stored as auditable evidence even
when it cannot safely be associated with a formula or used as a dose.
"""
from __future__ import annotations
import copy,re
import cv2,numpy as np
from app.schemas import Block,source_hash_of
from extract.formula_rows import _DOSE
from ocr.normalize import normalize_text
from ocr.recognize import recognize_crop,rec_model_name
from ocr.small_columns_components import propose_components_page
from ocr.small_columns_hybrid import _strong_mask
from ocr.vertical import imread_unicode,page_layout

_TAIL_NUMERIC=re.compile(r'^[\s]*[零〇一二三四五六七八九十百千两兩钱錢分厘毫半\d]')
_TRUSTED_TAIL=re.compile(r'^(?:黄土|去皮|去心|去核|去尖|酒炒|土炒|拌炒|炒|炙|炮|制|研|生用|打碎)+$')

def _read(image,box,scale):
 x,y,X,Y=map(int,box);crop=image[y:Y,x:X]
 if not crop.size:return '',0.0
 if scale!=1:crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
 return recognize_crop(np.ascontiguousarray(np.rot90(crop)),min_conf=.15)

def _ink(image,box,contrast):
 x,y,X,Y=map(int,box);crop=image[y:Y,x:X]
 if crop.size==0:return False
 gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
 mask=_strong_mask(gray,contrast)
 return int(mask.sum())>=12 and int(np.count_nonzero(mask.sum(axis=1)>=2))>=4

def _source(block,text,start=0,end=None):
 end=len(text) if end is None else end
 return {'normalized_range':[start,end],'text_norm':normalize_text(text[start:end]),'text':text[start:end],
         'sources':[{'block_ref':block.block_ref,'line_id':block.line_id,'bbox':list(block.bbox),
                     'normalized_range':[start,end],'block_raw_range':[start,end],
                     'raw_range':[start,end],'text':text[start:end],'mapping_precision':'crop'}]}

def attach_small_columns(ocr,formula_extraction,*,server_fallback=False):
 """Return detached evidence, modified copy of rows and diagnostics; never mutate input."""
 result=copy.deepcopy(formula_extraction);diagnostics={'enabled':True,'candidates':[],'accepted':0,'fallback':False}
 if ocr.is_demo or not ocr.model_loaded or not ocr.image_path or ocr.tables:
  diagnostics.update(fallback=True,reason='demo_missing_image_or_printed_table');return [],result,diagnostics
 image=imread_unicode(ocr.image_path)
 if image is None:diagnostics.update(fallback=True,reason='unreadable_image');return [],result,diagnostics
 cols=page_layout(ocr.image_path)['cols'];out=propose_components_page(image,cols,ocr.blocks)
 evidence=[];pending=[];block_map={b.block_ref:b for b in ocr.blocks}
 for p in out['predictions']:
  main=p['main_bbox'];sides={s['side']:s['bbox'] for s in p['columns']};scale=p['evidence']['page_typography'];trace={'parent_bbox':p['parent_bbox'],'main_bbox':main,'columns':p['columns'],'decision':'pending'}
  support=[b for b in ocr.blocks if b.bbox[0]<p['parent_bbox'][2] and b.bbox[2]>p['parent_bbox'][0] and b.bbox[1]<main[3] and b.bbox[3]>main[1]]
  if len(support)!=1:trace.update(decision='fallback',reason='nonunique_parent_ocr');diagnostics['candidates'].append(trace);continue
  parent=support[0];items={};reads={}
  for role,box in [('main',main),('right',sides['right']),('left',sides['left'])]:
   a,ca=_read(image,box,1);b,cb=_read(image,box,2);reads[role]={'native':a,'native_conf':ca,'double':b,'double_conf':cb,'ink':_ink(image,box,scale['ink_contrast'])}
   if role=='right' and server_fallback and (not _DOSE.match(normalize_text(b or '')) or min(ca,cb)<.8):
    # Keep mobile unless two server scales agree on a valid leading dose.
    reads[role]['mobile_before_server']={'native':a,'native_conf':ca,'double':b,'double_conf':cb}
    try:
     from ocr.small_columns_server import read_pair
     (sa,sca),(sb,scb)=read_pair(image,box)
     if _DOSE.match(normalize_text(sb or '')) and normalize_text(sa)==normalize_text(sb) and min(sca,scb)>=.8:
      a,ca,b,cb=sa,sca,sb,scb;reads[role].update(native=a,native_conf=ca,double=b,double_conf=cb,recognizer='official_server_fallback')
     else:reads[role]['server_status']='unstable_or_low_confidence'
    except Exception as exc:reads[role]['server_status']='unavailable_'+type(exc).__name__
   if not a and not b:continue
   text=b or a;conf=cb if b else ca;line='SC-'+source_hash_of(ocr.page_id,p['parent_bbox'],main,role,box)
   engine='PP-OCRv5_server_rec' if reads[role].get('recognizer')=='official_server_fallback' else rec_model_name()
   items[role]=Block(page_id=ocr.page_id,bbox=list(box),line_id=line,quote=text,text=text,text_norm=normalize_text(text),conf=conf,kind='small_column_evidence',source_hash=source_hash_of(ocr.page_id,box,text),evidence_role=role,evidence_status='待核',parent_block_ref=parent.block_ref,evidence_engine=engine,col_id=parent.col_id,reading_order=parent.reading_order)
  trace['recognition']=reads
  # Stable local readings are separately exposed for audit and fragment-level
  # measurement; their presence does not authorize assigning a dose.
  evidence.extend(items.values())
  m=reads['main'];r=reads['right'];norm_right=normalize_text(r['double'] or '');matched=_DOSE.match(norm_right)
  stable_main=(m['ink'] and min(m['native_conf'],m['double_conf'])>=.90 and bool(m['native']) and normalize_text(m['native'])==normalize_text(m['double']))
  stable_right=(r['ink'] and min(r['native_conf'],r['double_conf'])>=.80 and bool(r['native']) and normalize_text(r['native'])==normalize_text(r['double']))
  dose=r['double'][:matched.end()] if matched and len(norm_right)==len(r['double']) else ''
  tail=norm_right[matched.end():].strip() if matched else ''
  if not (stable_main and stable_right and dose and not dose.startswith('各') and not _TAIL_NUMERIC.match(tail) and (not tail or _TRUSTED_TAIL.fullmatch(tail))):
   trace.update(decision='fallback',reason='unconfirmed_main_or_complete_dose');diagnostics['candidates'].append(trace);continue
  candidates=[]
  for formula in result.get('formulas',[]):
   for row in formula.get('rows',[]):
    src=row.get('herb',{}).get('source') or {}; refs=[s.get('block_ref') for s in src.get('sources',[])]
    if parent.block_ref in refs and normalize_text(src.get('text',''))==normalize_text(m['double']):candidates.append(row)
  if len(candidates)!=1 or candidates[0]['dose'].get('status')!='待核':
   trace.update(decision='fallback',reason='nonunique_or_existing_dose');diagnostics['candidates'].append(trace);continue
  pending.append((candidates[0],items,dose,trace));diagnostics['candidates'].append(trace)
 # Multiple competing candidate regions for the same row must never win by order.
 for row,items,dose,trace in pending:
  if sum(other is row for other,_,_,_ in pending)!=1:
   trace.update(decision='fallback',reason='competing_candidates');continue
  mainblock,rightblock=items['main'],items['right'];mainblock.evidence_status='已识别';rightblock.evidence_status='已识别'
  row['herb']={'value':mainblock.text,'status':'已识别','source':_source(mainblock,mainblock.text)}
  row['dose']={'value':dose,'status':'已识别','source':_source(rightblock,rightblock.text,0,len(dose)),'shared':False}
  # Keep trailing and left-track text in original reading order; do not silently
  # correct or interpret uncertain preparation text as a verified fact.
  processing=[]
  if len(rightblock.text)>len(dose):
   processing.append({'role':'right_after_dose','text':rightblock.text[len(dose):],
                      'status':'待核','source':_source(rightblock,rightblock.text,len(dose))})
  if items.get('left'):
   leftblock=items['left'];processing.append({'role':'left','text':leftblock.text,
                                             'status':'待核','source':_source(leftblock,leftblock.text)})
  row['dose']['processing_context']={'status':'待核','fragments':processing}
  row['small_column_evidence']={'parent_block_ref':mainblock.parent_block_ref,'main':mainblock.block_ref,'right':rightblock.block_ref,'left':items.get('left').block_ref if items.get('left') else None,'left_status':'待核','reading_order':['main','right','left']}
  trace.update(decision='attached',dose=dose);diagnostics['accepted']+=1
 return evidence,result,diagnostics
