"""步骤 5 浏览器验证专用服务：隔离数据并预置清晰的受控交互样例。"""
import dataclasses
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.config import CONFIG
out=Path(sys.argv[1]).resolve();port=int(sys.argv[2]);paths=dataclasses.replace(CONFIG.paths,uploads_dir=out/'uploads',store_dir=out/'store',reports_dir=out/'reports',db_path=out/'store'/'tcm.sqlite3');object.__setattr__(CONFIG,'paths',paths);paths.ensure_dirs()
from PIL import Image,ImageDraw,ImageFont
from app.schemas import Block,OCRResult
from ocr.normalize import normalize_text
from app.main import app,_ingest
pid='P20261119000000-abc123';image_path=paths.uploads_dir/(pid+'.png')
parts=[('桂枝湯',[770,70,820,220]),('桂枝甘草',[650,130,700,320]),('各三',[530,130,580,230]),('錢',[460,130,510,185]),('共為細末',[350,130,400,320]),('稀涎散',[770,390,820,540]),('白礬生用牙皂角',[570,390,620,710]),('共為細末',[350,390,400,580]),('四君子湯',[770,750,820,940]),('人參二錢甘草三錢',[570,750,620,1110])]
im=Image.new('RGB',(900,1200),'#fcf4e5');draw=ImageDraw.Draw(im)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',32)
bs=[]
for n,(text,box) in enumerate(parts,1):
 x1,y1,x2,y2=box
 for i,ch in enumerate(text):draw.text((x1,y1+i*36),ch,font=font,fill='#201c17')
 bs.append(Block(page_id=pid,line_id=f'R{n:03}',block_ref=f'{pid}:R{n:03}',reading_order=n,col_id=n,text=text,text_norm=normalize_text(text),bbox=box,conf=.99,source_hash=f'fixture-{n}'))
im.save(image_path)
ocr=OCRResult(page_id=pid,image_path=str(image_path),width=900,height=1200,blocks=bs,tables=[],engine='controlled',model_loaded=False,is_demo=True,source_hash='stage5-fixture')
_ingest(ocr)
import uvicorn
uvicorn.run(app,host='127.0.0.1',port=port,log_level='warning')
