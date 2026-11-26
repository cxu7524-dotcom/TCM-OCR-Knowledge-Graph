# ============================================================
#  赛题二原型 · 一键启动（Windows / PowerShell 7+）
#
#  用法（在项目根目录或任意目录都行）：
#      pwsh -File scripts\start.ps1                # 默认 8100，占用则顺延 8000 / 8101-8103
#      pwsh -File scripts\start.ps1 -Port 8123     # 指定端口
#      pwsh -File scripts\start.ps1 -NoBrowser     # 不自动开浏览器（录屏 / 服务器上用）
#      pwsh -File scripts\start.ps1 -NoLlm         # 不走本地大模型，只用规则链路
#
#  说明：
#    - 优先用项目内 .venv，没有则退回系统 python（会在第一行打印用的是哪个）
#    - 核心推理全在本机：OCR 走 rapidocr-onnxruntime，问答走本机 Ollama（127.0.0.1）
#    - 没起 Ollama 不会崩，问答会自动降级并在界面提示「当前为字面检索」
# ============================================================
param(
    [int]$Port = 0,
    [switch]$NoBrowser,
    [switch]$NoLlm
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if ($NoLlm) {
    $env:TCM_LLM_BACKEND = "none"           # 显式要规则链路时，盖过可能已存在的环境变量
} elseif (-not $env:TCM_LLM_BACKEND) {
    $env:TCM_LLM_BACKEND = "ollama"
}
if (-not $env:TCM_LLM_MODEL) { $env:TCM_LLM_MODEL = "qwen2.5:7b" }
if ($NoBrowser) { $env:TCM_NO_BROWSER = "1" }

function Test-PortBusy([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $p)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

# ---- Python：项目 venv 优先 ----
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $sys = Get-Command python -ErrorAction SilentlyContinue
    if (-not $sys) {
        Write-Host "[ERROR] 没找到 Python：装 Python 3.10+ 并勾选 Add to PATH，或先在项目根目录建 .venv" -ForegroundColor Red
        exit 1
    }
    $python = $sys.Source
    Write-Host "  [note] 没有 .venv，改用系统 Python：$python"
}

if (-not (Test-Path (Join-Path $root "requirements.txt"))) {
    Write-Host "[ERROR] 当前目录不像项目根目录（缺 requirements.txt）：$root" -ForegroundColor Red
    exit 1
}

# ---- 端口：显式 > 环境变量 > 默认顺延 ----
if ($Port -gt 0) {
    $candidates = @($Port)
} elseif ($env:TCM_PORT) {
    $candidates = @([int]$env:TCM_PORT)
} else {
    $candidates = @(8100, 8000, 8101, 8102, 8103)
}
$chosen = 0
$skipped = @()
foreach ($p in $candidates) {
    if (Test-PortBusy $p) { $skipped += $p; continue }
    $chosen = $p
    break
}
if (-not $chosen) {
    Write-Host ("[ERROR] 端口 " + ($candidates -join " / ") + " 都被占用：先关掉占用的服务，或用 -Port 8123") -ForegroundColor Red
    exit 1
}
if ($skipped.Count -gt 0) {
    Write-Host ("  [note] " + ($skipped -join " / ") + " 已被占用，改用 $chosen")
}

Write-Host ""
Write-Host "  中医古籍 OCR 与知识图谱问答原型"
Write-Host "  地址：  http://127.0.0.1:$chosen"
Write-Host "  模型：  $($env:TCM_LLM_BACKEND) / $($env:TCM_LLM_MODEL)"
Write-Host "  页面：  / 四步演示台   /home 智能问答   /lounge 聊天室"
Write-Host "  声明：  仅供参考、非诊疗建议"
Write-Host ""

if (-not $env:TCM_NO_BROWSER) {
    try { Start-Process "http://127.0.0.1:$chosen" | Out-Null } catch { }
}

& $python "app\main.py" --port $chosen
exit $LASTEXITCODE
