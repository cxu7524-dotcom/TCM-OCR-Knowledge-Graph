"""Save browser screenshots immediately after clicking each P35 target field."""
import os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/small_columns_step6'
source=(out/'browser_extra_runner.py').read_text(encoding='utf8')
source=source.replace("for n in ('p35','wenyi12','wenyi08'):", "for n in ('p35',):")
needle="td.click(timeout=10000);entry['clicks'].append({'table':k,'row':int(rr),'column':int(cc),'field':'药材' if cc=='1' else '剂量','value':val,**selected(pg)})"
assert needle in source
source=source.replace(needle,needle+"\n       if n=='p35' and val in ('白术','五錢'):\n        pg.locator('#pageCanvas').screenshot(path=str(run/('p35_herb.png' if val=='白术' else 'p35_dose.png')))")
generated=out/'browser_p35_runner.py';generated.write_text(source,encoding='utf8')
env=os.environ.copy();env.update(STEP6_MODE='p35_screens_on',TCM_OCR_SMALL_COLUMNS='1',TCM_OCR_SMALL_COLUMNS_SERVER='1')
subprocess.run([sys.executable,str(generated)],cwd=root,env=env,check=True)
