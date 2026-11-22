"""仅供步骤 4 验证：在指定隔离目录启动真实 FastAPI 服务。"""
import dataclasses
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import CONFIG
out=Path(sys.argv[1]).resolve();port=int(sys.argv[2])
paths=dataclasses.replace(CONFIG.paths,uploads_dir=out/'uploads',store_dir=out/'store',
                          reports_dir=out/'reports',db_path=out/'store'/'tcm.sqlite3')
object.__setattr__(CONFIG,'paths',paths)
paths.ensure_dirs()
import uvicorn
from app.main import app
uvicorn.run(app,host='127.0.0.1',port=port,log_level='warning')
