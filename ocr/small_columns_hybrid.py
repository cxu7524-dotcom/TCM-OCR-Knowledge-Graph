"""OCR-assisted boundary evidence for geometric small-column proposals.

This is an offline geometry experiment. It does not change production OCR, write
recognized text into results, or parse formulas. OCR failures yield fallback.
"""
from __future__ import annotations
import cv2
import numpy as np
from .small_columns import propose_page
from .recognize import recognize_crop
from .normalize import normalize_text


def _box(block):
    return block.get('bbox') if isinstance(block,dict) else getattr(block,'bbox',None)


def _field(block,name,default=''):
    return block.get(name,default) if isinstance(block,dict) else getattr(block,name,default)


def _supported_blocks(parent,main,blocks):
    x1,y1,x2,y2=parent; mx1,my1,mx2,my2=main;out=[]
    for block in blocks:
        b=_box(block)
        if not b or len(b)!=4:continue
        horizontal=max(0,min(x2,b[2])-max(x1,b[0]))
        vertical=max(0,min(my2,b[3])-max(my1,b[1]))
        if horizontal>=.4*max(1,x2-x1) and vertical>=.35*max(1,my2-my1):
            out.append({'line_id':_field(block,'line_id'),'bbox':list(b),'confidence':_field(block,'conf',0)})
    return out


def _gray(image,box):
    x1,y1,x2,y2=map(int,box);part=image[y1:y2,x1:x2]
    if part.size==0:return None
    return cv2.cvtColor(part,cv2.COLOR_BGR2GRAY) if part.ndim==3 else part


def _strong_mask(gray,contrast):
    bg=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)
    return ((bg-gray.astype(np.int16))>=max(18.,min(28.,contrast*.4))).astype(np.uint8)


def _main_link_evidence(image, proposal):
    """Disallow a skipped independent strong-ink band or an excessive gap."""
    main=proposal['main_bbox']; children=proposal['columns']; parent=proposal['parent_bbox']
    y0=int(main[3]);y1=int(min(c['bbox'][1] for c in children));scale=proposal['evidence']['page_typography']
    if y1<y0:return {'valid':False,'reason':'child_before_main_end','gap':y1-y0}
    if y1-y0>max(8,int(scale['height']*.75)):
        return {'valid':False,'reason':'nonlocal_main_gap','gap':y1-y0}
    if y1-y0<=4:return {'valid':True,'reason':'adjacent','gap':y1-y0,'intervening_components':[]}
    gray=_gray(image,[parent[0],max(0,y0-12),parent[2],min(image.shape[0],y1+12)])
    if gray is None:return {'valid':False,'reason':'parent_crop_unreadable','gap':y1-y0}
    mask=_strong_mask(gray,scale['ink_contrast']);v0=y0+2-max(0,y0-12);v1=y1-2-max(0,y0-12)
    band=mask[max(0,v0):max(0,v1)]
    if band.size==0:return {'valid':True,'reason':'adjacent','gap':y1-y0,'intervening_components':[]}
    _,_,stats,_=cv2.connectedComponentsWithStats(band,8)
    minimum=max(12,int(scale['height']*scale['height']*.035))
    found=[{'x':int(x+parent[0]),'y':int(y+y0+2),'width':int(w),'height':int(h),'ink':int(n)} for x,y,w,h,n in stats[1:] if n>=minimum and h>=3]
    return {'valid':not bool(found),'reason':'intervening_ink' if found else 'short_clean_gap','gap':y1-y0,'intervening_components':found}


def _safe_end(image, proposal, side, step):
    """Bound candidate extensions at a local blank gap or the page/column edge."""
    parent=proposal['parent_bbox'];crop=proposal['crop_bbox'];b=side['bbox'];end=int(b[3]);height=image.shape[0]
    max_end=min(height,int(crop[3]),end+step)
    gray=_gray(image,[crop[0],max(0,end-12),crop[2],min(height,max_end+6)])
    if gray is None:return end,'unreadable'
    mask=_strong_mask(gray,proposal['evidence']['page_typography']['ink_contrast']);origin=max(0,end-12)
    for y in range(end,max_end+1):
        # Two blank rows mark separation: do not enter the following text block.
        i=y-origin
        if i+1<len(mask) and int(mask[i].sum())<=1 and int(mask[i+1].sum())<=1:
            return min(max_end,y+1),'two_blank_rows'
    return end,'no_proven_boundary'


def _recognize(image, box, side):
    x1,y1,x2,y2=map(int,box);crop=image[y1:y2,x1:x2]
    if crop.size==0:return '',0.0
    if side=='right':crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    return recognize_crop(np.ascontiguousarray(np.rot90(crop)),min_conf=.15)


def _refine_side(image, proposal, side):
    box=list(map(int,side['bbox']));step=max(2,min(8,int(round(proposal['evidence']['page_typography']['gap']))))
    max_end,stop=_safe_end(image,proposal,side,step)
    ends=[('shortened',max(box[1]+4,box[3]-step)),('original',box[3]),('extended',max_end)]
    trace=[]
    for label,end in ends:
        b=[box[0],box[1],box[2],end]
        if label=='extended' and end<=box[3]:
            trace.append({'variant':label,'bbox':b,'status':'blocked','reason':stop});continue
        text,conf=_recognize(image,b,side['side'])
        trace.append({'variant':label,'bbox':b,'text':text,'confidence':conf,'status':'read' if text else 'unreadable'})
    original=trace[1];extended=trace[2];selected='original';reason='conservative_original'
    if extended['status']=='read' and original['status']=='read':
        norm0=normalize_text(original['text']);norm1=normalize_text(extended['text'])
        # Recognizer confidence alone cannot move the border. There must be actual
        # ink in the extension and the same or a clearly continued reading.
        xa,ya,xb,yb=extended['bbox'];extra=_gray(image,[xa,box[3],xb,yb]);prior=_gray(image,[xa,max(box[1],box[3]-12),xb,box[3]])
        extra_ink=int(_strong_mask(extra,proposal['evidence']['page_typography']['ink_contrast']).sum()) if extra is not None and min(extra.shape)>=3 else 0
        prior_ink=int(_strong_mask(prior,proposal['evidence']['page_typography']['ink_contrast']).sum()) if prior is not None and min(prior.shape)>=3 else 0
        stable=bool(norm0 and norm1 and (norm0==norm1 or (len(norm1)>len(norm0) and norm1.startswith(norm0))))
        if stable and extra_ink>=3 and prior_ink>=3 and extended['confidence']>=max(.6,original['confidence']+.10):
            selected='extended';reason='aligned_ink_stable_reading_and_confidence_gain'
    chosen=next(x for x in trace if x['variant']==selected)
    return {'side':side['side'],'bbox':chosen['bbox']}, {'side':side['side'],'selection':selected,'reason':reason,'stop':stop,'candidates':trace}


def propose_hybrid_page(image, columns, ocr_blocks):
    """Return only geometric proposals plus separate diagnostic OCR evidence."""
    if image is None or not ocr_blocks:return {'predictions':[],'audit':[{'reason':'missing_auto_ocr_fallback'}]}
    geometry=propose_page(image,columns);predictions=[];audit=[]
    for p in geometry:
        support=_supported_blocks(p['parent_bbox'],p['main_bbox'],ocr_blocks)
        link=_main_link_evidence(image,p)
        event={'parent_bbox':p['parent_bbox'],'geometric_main':p['main_bbox'],'geometric_columns':p['columns'],'auto_ocr_support':support,'link_evidence':link}
        if not support or not link['valid']:
            event['decision']='fallback_outer';event['reason']='no_matching_auto_ocr' if not support else link['reason'];audit.append(event);continue
        sides=[];side_audit=[]
        for side in p['columns']:
            chosen,ev=_refine_side(image,p,side);sides.append(chosen);side_audit.append(ev)
        event['side_comparisons']=side_audit
        if any(s['candidates'][1]['status']!='read' for s in side_audit):
            event['decision']='fallback_outer';event['reason']='local_ocr_unreadable';audit.append(event);continue
        q=dict(p);q['columns']=sides;q['evidence']=dict(p['evidence'],hybrid_boundary=True,auto_ocr_line_ids=[x['line_id'] for x in support]);predictions.append(q)
        event['decision']='proposed';audit.append(event)
    return {'predictions':predictions,'audit':audit}
