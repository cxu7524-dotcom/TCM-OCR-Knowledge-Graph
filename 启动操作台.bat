@echo off
rem ============================================================
rem  赛题二 · 中医古籍 OCR 与知识图谱问答原型 · 一键启动
rem  1. 双击本文件；2. 浏览器自动打开 http://127.0.0.1:8000
rem  3. 保持窗口开启即为服务运行，关闭窗口即停止服务
rem  环境要求：Python 3.10+，已执行 pip install -r requirements.txt
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set "PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if exist "%PY%" goto have_py
set "PY=python"
where python >nul 2>&1 || goto no_python

:have_py
echo.
echo   赛题二原型启动中 ...
echo   页面地址： http://127.0.0.1:8000
echo   演示模式： 示例数据 / 模型未接入（指标一律待测）
echo   免责声明： 仅供参考、非诊疗建议
echo.

if not "%TCM_NO_BROWSER%"=="1" start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8000"

"%PY%" app\main.py --port 8000
pause
exit /b

:no_python
echo [错误] 未找到 Python。请安装 Python 3.10+ 并勾选 Add python.exe to PATH。
pause
exit /b 1
