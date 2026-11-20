"""Conservative geometric proposals for small parallel notes in detected vertical columns.

No OCR, ground truth, or table integration. Coordinates use source-image pixels.
An empty proposal list means that the existing outer-column path must be used.
"""
from __future__ import annotations
import cv2
import numpy as np

PADDING = 7
WINDOW = 28
STEP = 4
MAX_VALLEY_RATIO = 0.10
MIN_HALF_INK = 18
MIN_WINDOWS = 2


def propose_in_column(image, bbox, x_limits=None, page_scale=None):
    """Generate dual-column proposals inside an automatically detected outer column.

    x_limits clip padding to neighboring detected columns. Returned parent bbox is the
    input detector bbox; all other boxes are source-image coordinates.
    """
    if image is None or image.size == 0 or len(bbox) != 4:
        return []
    h,w=image.shape[:2];x1,y1,x2,y2=map(int,bbox)
    lo,hi=x_limits if x_limits is not None else (0,w)
    a=max(0,int(lo),x1-PADDING);z=min(w,int(hi),x2+PADDING)
    top=max(0,y1,int(h*.20));bottom=min(h,y2,int(h*.92))
    if z-a<20 or bottom-top<WINDOW or x1 < w*.03 or x2 > w*.97:return []
    gray=cv2.cvtColor(image[top:bottom,a:z],cv2.COLOR_BGR2GRAY) if image.ndim==3 else image[top:bottom,a:z]
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,14)//255
    hits=[];width=z-a
    for y in range(0,len(bw)-WINDOW+1,STEP):
        prof=bw[y:y+WINDOW].sum(axis=0);smooth=np.convolve(prof,[.25,.5,.25],'same');best=None
        for k in range(int(width*.30),int(width*.70)):
            lp=smooth[max(0,k-15):k-3];rp=smooth[k+3:min(width,k+16)]
            if not len(lp) or not len(rp):continue
            left_peak=float(lp.max());right_peak=float(rp.max())
            left_ink=int(prof[max(0,k-18):k-2].sum());right_ink=int(prof[k+2:min(width,k+18)].sum())
            if min(left_peak,right_peak)<4 or min(left_ink,right_ink)<MIN_HALF_INK:continue
            ratio=float(smooth[k]/min(left_peak,right_peak))
            if ratio<=MAX_VALLEY_RATIO and (best is None or ratio<best[0]):best=(ratio,k)
        if best:hits.append((y,best[1],best[0]))
    groups=[]
    for hit in hits:
        if groups and hit[0]-groups[-1][-1][0]<=STEP*2 and abs(hit[1]-groups[-1][-1][1])<=4:groups[-1].append(hit)
        else:groups.append([hit])
    proposals=[]
    for group in groups:
        if len(group)<MIN_WINDOWS or (len(group)<5 and min(v[2] for v in group)>0.035):continue
        k=int(round(float(np.median([x[1] for x in group])))); start=group[0][0];end=group[-1][0]+WINDOW
        # Stage 1 is a core only. Stage 2 traces the two ink tracks separately.
        details = _trace_boundaries(gray, bw, k, start, end, x2-x1)
        if details is None:
            continue
        main_top, main_bottom, lanes, evidence = details
        # Use page-level normal character geometry to recover complete preceding glyph bands.
        recovered = _recover_main_bands(gray, bw, x1-a, x2-a, lanes, page_scale)
        if recovered is None:
            continue  # insufficient evidence: retain the outer-column path
        main_top, main_bottom, band_evidence = recovered
        lanes = [(max(y0, main_bottom), y1) for y0,y1 in lanes]
        if any(y1-y0<3 for y0,y1 in lanes):
            continue
        evidence.update(band_evidence)
        boxes = [[a+k,top+lanes[0][0],z,top+lanes[0][1]],
                 [a,top+lanes[1][0],a+k,top+lanes[1][1]]]
        proposals.append({'parent_bbox':[x1,y1,x2,y2],'crop_bbox':[a,top,z,bottom],
                          'main_bbox':[x1,top+main_top,x2,top+main_bottom],
                          'columns':[{'side':'right','bbox':boxes[0]},{'side':'left','bbox':boxes[1]}],
                          'reading_order':['main','right','left'],
                          'evidence':{'windows':len(group),'valley_x':a+k,'min_valley_ratio':round(min(v[2] for v in group),4),'core_y':[top+start,top+end],**evidence}})
    return proposals


def _runs(mask):
    edges=np.diff(np.r_[0,mask.astype(np.uint8),0].astype(np.int16))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def _trace_boundaries(gray, bw, split, core_start, core_end, main_width):
    """Trace each lane separately and stop at the next full-width main glyph."""
    height,width=bw.shape
    rows=(bw.sum(axis=1)>=3).astype(np.uint8)
    rows=cv2.morphologyEx(rows[:,None],cv2.MORPH_CLOSE,np.ones((11,1),np.uint8)).ravel()
    main_end=max(0,core_start-2);cursor=main_end
    while cursor>0 and not rows[cursor-1] and main_end-cursor<12:cursor-=1
    main_bottom=cursor
    while cursor>0 and rows[cursor-1] and main_bottom-cursor<95:cursor-=1
    main_top=cursor
    if not 18<=main_bottom-main_top<=95 or core_end-core_start>150:return None
    # Full-width ink spanning the seam is evidence of the next main character.
    stop_rows=(bw.sum(axis=1)>=3).astype(np.uint8)
    stop_rows=cv2.morphologyEx(stop_rows[:,None],cv2.MORPH_CLOSE,np.ones((5,1),np.uint8)).ravel()
    next_main=height
    found_at=None
    for y in range(max(core_start+25,core_end-12),min(height-20,core_start+200),2):
        prof=bw[y:y+20].sum(axis=0)
        left=int(prof[:max(0,split-3)].sum());right=int(prof[min(width,split+3):].sum())
        crossing=int(prof[max(0,split-2):min(width,split+3)].sum())
        if min(left,right)>=35 and crossing>=.7*min(left,right):
            found_at=y
            cursor=y+10
            while cursor>max(core_end-5,y-12) and stop_rows[cursor-1]:cursor-=1
            next_main=cursor
            break
    # Locate the first short band with two active tracks and a clear central gap.
    begin=core_start
    for y in range(max(0,core_start-10),min(height-10,core_start+20),2):
        prof=bw[y:y+10].sum(axis=0)
        left=int(prof[:max(0,split-2)].sum())
        right=int(prof[min(width,split+2):].sum())
        seam=int(prof[max(0,split-1):min(width,split+2)].sum())
        if min(left,right)>=10 and seam<=.1*min(left,right)+1:
            begin=y
            break
    main_bottom=begin
    background=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)
    contrast=background-gray.astype(np.int16)
    strong=(contrast>=22).astype(np.uint8)
    # Stop before a near-full-width printed page rule.
    line_rows=np.flatnonzero(bw.sum(axis=1)>=width*.60)
    later=[int(y) for y in line_rows if y>core_start+20]
    if later:next_main=min(next_main,max(core_start+20,min(later)-1))
    # Page rules are long horizontal lines, not single-character strokes.
    rules=cv2.morphologyEx(strong,cv2.MORPH_OPEN,np.ones((1,max(12,int(width*.8))),np.uint8))
    strong=np.maximum(strong-rules,0)
    lanes=[]
    for xs,xe in ((split,width),(0,split)):
        if xe-xs<5:return None
        lane=strong[:,xs:xe];active=(lane.sum(axis=1)>=2).astype(np.uint8)
        active=cv2.morphologyEx(active[:,None],cv2.MORPH_CLOSE,np.ones((7,1),np.uint8)).ravel()
        spans=[]
        for a,b in _runs(active):
            if b<=core_start-8 or a>=next_main:continue
            vals=contrast[a:b,xs:xe]; pixels=lane[a:b];count=int(pixels.sum())
            if count<10:continue
            spans.append((a,min(b,next_main),count))
        starts=[i for i,(a,b,_) in enumerate(spans) if a<=core_start+24 and b>core_start-8]
        if not starts:return None
        i=starts[0];y0,y1,_=spans[i];y0=max(y0,begin)
        for a,b,_ in spans[i+1:]:
            if a-y1>22:break
            y1=b
        if y1-y0<3 or y0>=next_main:return None
        lanes.append((max(0,int(y0)-1),min(int(next_main),int(y1)+4)))
    return int(main_top),int(main_bottom),lanes,{'next_main_y':int(next_main),'blocker_window_y':found_at,'contrast_floor':22}


def estimate_page_typography(image, columns):
    """Estimate normal glyph height, ink width and inter-glyph gap from page columns."""
    h,w=image.shape[:2]; heights=[]; widths=[]; gaps=[]; contrasts=[]
    nominal=[int(c['bbox'][2]-c['bbox'][0]) for c in columns
             if c['bbox'][0]>w*.04 and c['bbox'][2]<w*.96 and 18<=c['bbox'][2]-c['bbox'][0]<=h*.10]
    if not nominal:
        return None
    median_width=float(np.median(nominal))
    for col in columns:
        x1,_,x2,_=map(int,col['bbox']);wid=x2-x1
        if not (x1>w*.04 and x2<w*.96 and median_width*.72<=wid<=median_width*1.35):
            continue
        band=image[int(.24*h):int(.89*h),x1:x2]
        if band.size==0:continue
        gray=cv2.cvtColor(band,cv2.COLOR_BGR2GRAY) if band.ndim==3 else band
        bg=cv2.GaussianBlur(gray,(31,31),0).astype(np.int16)
        delta=bg-gray.astype(np.int16)
        bw=(delta>=14).astype(np.uint8)
        active=(bw.sum(axis=1)>=max(3,int(wid*.11))).astype(np.uint8)
        active=cv2.morphologyEx(active[:,None],cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)).ravel()
        good=[]
        for a,b in _runs(active):
            hh=int(b-a)
            if not 10<=hh<=int(wid*1.4):continue
            ink=bw[a:b].sum(axis=0);xs=np.flatnonzero(ink>=2)
            if not len(xs) or xs[-1]-xs[0]+1<wid*.5:continue
            heights.append(hh);widths.append(int(xs[-1]-xs[0]+1));good.append((a,b))
            vals=delta[a:b][bw[a:b]>0]
            if len(vals):contrasts.append(float(np.median(vals)))
        for (_,end),(start,_) in zip(good,good[1:]):
            if 1<=start-end<=16:gaps.append(int(start-end))
    if len(heights)<10:
        return None
    return {'height':float(np.median(heights)),'width':float(np.median(widths)),
            'gap':float(np.median(gaps)) if gaps else median_width*.15,
            'ink_contrast':float(np.median(contrasts)) if contrasts else 40.0,
            'sample_bands':len(heights)}


def _recover_main_bands(gray, bw, x1, x2, lanes, scale):
    """Recover complete large-character bands immediately before the two small tracks.

    Only ink and page-derived geometry are used. A large blank or a smaller preceding
    band terminates the walk; uncertain layouts return None instead of forcing a split.
    """
    if not scale:return None
    hh=float(scale['height']);ww=float(scale['width']);gap=float(scale['gap'])
    x1=max(0,x1);x2=min(bw.shape[1],x2);parent=bw[:,x1:x2]
    if parent.size==0:return None
    y_note=min(y0 for y0,_ in lanes)
    activity=(parent.sum(axis=1)>=max(3,int((x2-x1)*.11))).astype(np.uint8)
    activity=cv2.morphologyEx(activity[:,None],cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)).ravel()
    intervals=_runs(activity);good=[]
    for i,(a,b) in enumerate(intervals):
        # A run can contain several large characters joined to the annotation.
        # Limit that long run at the already detected two-track transition.
        if b-a>hh*2.8 and a<y_note<b:
            b=min(b,y_note+2)
        length=b-a
        if not (hh*.35<=length<=hh*2.8 and a<y_note and b<=y_note+hh*.6):continue
        prof=parent[a:b].sum(axis=0);xs=np.flatnonzero(prof>=2)
        if not len(xs) or xs[-1]-xs[0]+1 < ww*.65:continue
        good.append((i,int(a),int(b)))
    if not good:return None
    idx,first,last=good[-1]; chosen=[good[-1]]
    max_gap=max(7,int(round(gap*2.5))); max_span=int(round(hh*4.3+gap*3))
    for candidate in reversed(good[:-1]):
        ci,a,b=candidate
        if first-b>max_gap or last-a>max_span:break
        # An intervening substantial narrow band indicates a prior annotation.
        intervening=intervals[ci+1:idx]
        if any((ib-ia)>=hh*.45 and int(parent[ia:ib].sum())>=30 for ia,ib in intervening):break
        chosen.insert(0,candidate);idx=ci;first=a
    # Include small separated strokes of first or last large glyph, but never cross
    # a full-height annotation track or a gap larger than the estimated spacing.
    for a,b in reversed(intervals[:idx]):
        if first-b>max_gap or last-a>max_span:break
        if b-a>hh*.42:break
        if int(parent[a:b].sum())>=8:first=int(a)
    for a,b in intervals[chosen[-1][0]+1:]:
        if a>y_note or a-last>max_gap or b>y_note+2:break
        if b-a>hh*.42:break
        if int(parent[a:b].sum())>=8:last=int(b)
    # Recover low-contrast edge strokes against the page's ordinary ink contrast.
    # Restrict this to a narrow margin next to the confirmed large-glyph band.
    local_gray=gray[:,x1:x2]
    background=cv2.GaussianBlur(local_gray,(31,31),0).astype(np.int16)
    threshold=max(8.0,min(16.0,float(scale['ink_contrast'])*.28))
    weak=(background-local_gray.astype(np.int16))>=threshold
    top=first
    for y in range(first-1,max(-1,first-4),-1):
        if int(weak[y].sum())<2:break
        top=y
    top=max(0,top-1)
    bottom=min(parent.shape[0],last+1)
    for y in range(bottom,min(parent.shape[0],last+3,y_note)):
        if int(weak[y].sum())<2:break
        bottom=y+1
    if bottom-top<hh*.65 or bottom>y_note+hh*.6:return None
    return top,bottom,{'page_typography':scale,'main_bands':[[int(a),int(b)] for _,a,b in chosen],
                      'main_band_y_origin':'crop_bbox top','main_boundary':'page-scale bands and adjacent short strokes',
                      'weak_contrast_threshold':round(threshold,2)}


def propose_page(image, columns):
    """Run on detected page-layout columns; do not borrow annotated coordinates."""
    result=[];w=image.shape[1]
    scale=estimate_page_typography(image,columns)
    for i,col in enumerate(columns):
        bbox=col['bbox'];x1,_,x2,_=bbox
        left=max([b['bbox'][2] for j,b in enumerate(columns) if j!=i and b['bbox'][2]<=x1],default=0)
        right=min([b['bbox'][0] for j,b in enumerate(columns) if j!=i and b['bbox'][0]>=x2],default=w)
        result.extend(propose_in_column(image,bbox,((left+x1)//2,(right+x2)//2),scale))
    return result
