"""Repeat browser click checks on regression and frozen independent pages."""
import os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/small_columns_step6'
assert (out/'extra_manifest.json').exists(), 'freeze independent page before running'
source=(out/'browser_runner.py').read_text(encoding='utf8')
source=source.replace("for n in (37,39,40,41,50):", "for n in ('p35','wenyi12','wenyi08'):")
source=source.replace("img=ROOT/'古籍评测集'/'医家类'/'思濟堂方書'/f'page_{n}.png';pid=upload(pg,img)", "img=({'p35':ROOT/'古籍评测集/医家类/思濟堂方書/page_35.png','wenyi12':ROOT/'古籍评测集/医家类/温疫方論/page_12.png','wenyi08':ROOT/'古籍评测集/医家类/温疫方論/page_8.png'})[n];pid=upload(pg,img)")
generated=out/'browser_extra_runner.py';generated.write_text(source,encoding='utf8')
env=os.environ.copy();env.update(STEP6_MODE='extra_on',TCM_OCR_SMALL_COLUMNS='1',TCM_OCR_SMALL_COLUMNS_SERVER='1')
subprocess.run([sys.executable,str(generated)],cwd=root,env=env,check=True)
