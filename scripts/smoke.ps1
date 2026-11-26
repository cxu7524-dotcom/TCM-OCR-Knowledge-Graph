# ============================================================
#  赛题二原型 · 自检（服务起来后跑一遍，确认关键接口与页面都在）
#
#  用法：
#      powershell -NoProfile -File scripts\smoke.ps1              # 默认查 8100
#      powershell -NoProfile -File scripts\smoke.ps1 -Port 8000
#
#  输出：每项一行 PASS / FAIL；全通过退出码 0，否则 1。
#  只读 GET，不改数据、不跑模型。
# ============================================================
param(
    [int]$Port = 8100
)

$ErrorActionPreference = "Continue"
$base = "http://127.0.0.1:$Port"
$results = @()

function Add-Result([string]$name, [bool]$ok, [string]$detail) {
    $script:results += [pscustomobject]@{ Name = $name; Ok = $ok; Detail = $detail }
}

# Windows PowerShell 5.1 会把没有 charset 的 JSON 按 8859-1 解码，中文会乱码；
# 统一从原始字节按 UTF-8 解，再转对象。
function ConvertFrom-Utf8Json([byte[]]$bytes) {
    return ([Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json)
}

function Invoke-Json([string]$path) {
    $r = Invoke-WebRequest -Uri "$base$path" -UseBasicParsing -TimeoutSec 20
    try { return (ConvertFrom-Utf8Json $r.RawContentStream.ToArray()) } catch { }
    return ($r.Content | ConvertFrom-Json)
}

# ---- 1. 三个页面 ----
foreach ($page in @(@("/", "四步演示台"), @("/home", "智能问答"), @("/lounge", "聊天室"))) {
    try {
        $r = Invoke-WebRequest -Uri "$base$($page[0])" -UseBasicParsing -TimeoutSec 20
        $ok = ($r.StatusCode -eq 200 -and $r.RawContentLength -gt 20000)
        Add-Result ("页面 " + $page[0] + "（" + $page[1] + "）") $ok ("HTTP " + $r.StatusCode + " · " + [int]($r.RawContentLength / 1024) + " KB")
    } catch {
        Add-Result ("页面 " + $page[0] + "（" + $page[1] + "）") $false $_.Exception.Message
    }
}

# ---- 2. 评测指标 ----
try {
    $m = Invoke-Json "/api/metrics"
    if ($m.available) {
        $n = @($m.items).Count
        Add-Result "评测指标 /api/metrics" ($n -ge 4) ("$n 项 · 测试页 " + $m.test_pages)
    } else {
        Add-Result "评测指标 /api/metrics" $false ("尚未评测：" + $m.notice)
    }
} catch {
    Add-Result "评测指标 /api/metrics" $false $_.Exception.Message
}

# ---- 3. 检索索引 ----
try {
    $st = Invoke-Json "/api/rag/status"
    $idx = $st.index
    $vec = if ($null -ne $idx) { $idx.vectors } else { $st.vectors }
    $stale = if ($null -ne $idx) { $idx.stale } else { $st.stale }
    $health = if ($null -ne $idx) { $idx.health } else { $null }
    $detail = "$vec 段向量"
    if ($null -ne $stale) { $detail += " · stale=$stale" }
    if ($health) { $detail += " · " + $health }
    if ($st.enabled -eq $false) { $detail += " · RAG 已关闭（其余功能照常）" }
    Add-Result "语义索引 /api/rag/status" ($vec -gt 0) $detail
} catch {
    Add-Result "语义索引 /api/rag/status" $false $_.Exception.Message
}

# ---- 4. 聊天室 ----
try {
    $rooms = Invoke-Json "/api/rooms"
    $official = @($rooms.official).Count
    $guest = @($rooms.guest).Count
    Add-Result "聊天室 /api/rooms" ($official -ge 1) ("官方 " + $official + " 间 · 游客自建 " + $guest + " 间")
} catch {
    Add-Result "聊天室 /api/rooms" $false $_.Exception.Message
}

# ---- 5. 问答链路（空问题应被拒答，不跑模型） ----
try {
    $body = @{ question = "" } | ConvertTo-Json
    $r = Invoke-WebRequest -Uri "$base/api/qa" -Method POST -UseBasicParsing -TimeoutSec 30 `
        -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
    $ans = try { ConvertFrom-Utf8Json $r.RawContentStream.ToArray() } catch { $r.Content | ConvertFrom-Json }
    Add-Result "问答 /api/qa（空问题应拒答）" ([bool]$ans.refused) ("refused=" + $ans.refused + " · backend=" + $ans.backend)
} catch {
    Add-Result "问答 /api/qa（空问题应拒答）" $false $_.Exception.Message
}

# ---- 汇总 ----
Write-Host ""
Write-Host "  自检目标：$base"
Write-Host ""
foreach ($r in $results) {
    $tag = if ($r.Ok) { "PASS" } else { "FAIL" }
    $color = if ($r.Ok) { "Green" } else { "Red" }
    Write-Host ("  [{0}] {1,-34} {2}" -f $tag, $r.Name, $r.Detail) -ForegroundColor $color
}
$bad = @($results | Where-Object { -not $_.Ok }).Count
Write-Host ""
if ($bad -eq 0) {
    Write-Host ("  全部通过（" + $results.Count + " 项）") -ForegroundColor Green
    exit 0
}
Write-Host ("  有 " + $bad + " 项失败，共 " + $results.Count + " 项") -ForegroundColor Red
exit 1
