"""Step 3 offline candidate: geometric bands plus crossing ink components.

Recognition results are diagnostic only and never remove an otherwise supported
geometry candidate. This module is not connected to the production pipeline.
"""
import cv2
import numpy as np
from .small_columns import propose_page
from .small_columns_hybrid import _supported_blocks, _main_link_evidence, _recognize

def _crossing_end(mask, box):
    x,y,X,Y=map(int,box);bottom=min(mask.shape[0],Y+9);top=max(y,Y-20)
    tile=mask[top:bottom,x:X].astype('uint8')
    if not tile.size:return list(box),[]
    _,_,stats,_=cv2.connectedComponentsWithStats(tile,8);cross=[]
    for a,b,w,h,n in stats[1:]:
        start=int(top+b);end=int(top+b+h)
        if start<Y and Y<end<=Y+8 and n>=5 and end<bottom:
            cross.append({'bbox':[int(x+a),start,int(x+a+w),end],'pixels':int(n)})
    return [x,y,X,max([Y]+[c['bbox'][3] for c in cross])],cross

def propose_components_page(image,columns,ocr_blocks):
    if image is None or not ocr_blocks:return {'predictions':[],'audit':[{'reason':'missing_auto_ocr_fallback'}]}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    blur=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)
    mask=(blur-gray.astype(np.int16))>=22
    predictions=[];audit=[]
    for p in propose_page(image,columns):
        support=_supported_blocks(p['parent_bbox'],p['main_bbox'],ocr_blocks)
        link=_main_link_evidence(image,p)
        ev={'parent_bbox':p['parent_bbox'],'geometric_main':p['main_bbox'],'geometric_columns':p['columns'],'auto_ocr_support':support,'link_evidence':link}
        if not support or not link['valid']:
            ev.update(decision='fallback_outer',reason='no_matching_auto_ocr' if not support else link['reason']);audit.append(ev);continue
        sides=[];trace=[]
        for c in p['columns']:
            b,components=_crossing_end(mask,c['bbox']);sides.append({'side':c['side'],'bbox':b})
            text,conf=_recognize(image,c['bbox'],c['side'])
            trace.append({'side':c['side'],'crossing_components':components,'original_recognition':{'status':'read' if text else 'unreadable','text':text,'confidence':conf}})
        q=dict(p);q['columns']=sides;q['evidence']=dict(p['evidence'],component_boundary=True,auto_ocr_line_ids=[x['line_id'] for x in support]);predictions.append(q)
        ev.update(decision='proposed',side_evidence=trace,recognition_status_is_not_geometry_gate=True);audit.append(ev)
    return {'predictions':predictions,'audit':audit}
