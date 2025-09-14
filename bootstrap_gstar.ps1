<#
bootstrap_gstar_compat.ps1  (Windows PowerShell 5.1 compatible)
- 创建/更新 Conda 环境 (默认: gstar310, Python 3.10)
- 安装 requirements.txt
- 如无 .env 则从 .env.example 生成（先不用改也行）
- 拷贝 toy_corpus.jsonl 到项目目录
- 可选：如果找到 gstar_astar_proto.py，则跑一次冒烟测试

用法（在项目根目录的 PowerShell 里）：
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\bootstrap_gstar_compat.ps1 -ProjectDir "$($PWD.Path)" -MainPy "gstar_astar_proto.py"
#>

param(
  [string]$EnvName = "gstar310",
  [string]$PythonVersion = "3.10",
  [string]$ProjectDir = (Get-Location).Path,
  [string]$MainPy = "gstar_astar_proto.py"
)

function Fail([string]$msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

Write-Host "=== GStar Bootstrap (PowerShell 5.1) ===" -ForegroundColor Cyan
Write-Host "Env: $EnvName  Python: $PythonVersion"
Write-Host "ProjectDir: $ProjectDir"
Write-Host "Main script: $MainPy"
Write-Host ""

# --- 定位 conda 可执行文件 ---
$condaExe = $env:CONDA_EXE
if (-not $condaExe) {
  $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
  if ($cmd) { $condaExe = $cmd.Path }
}
if (-not $condaExe) {
  Fail "Conda 未找到。请先安装 Miniconda/Anaconda 并重新打开 PowerShell。"
}
Write-Host "Using Conda at: $condaExe"

# --- 路径 ---
$req        = Join-Path $PSScriptRoot "requirements.txt"
$envExample = Join-Path $PSScriptRoot ".env.example"
$dotEnv     = Join-Path $ProjectDir ".env"
$toy        = Join-Path $PSScriptRoot "toy_corpus.jsonl"
$main       = Join-Path $ProjectDir $MainPy

if (-not (Test-Path $req)) { Fail "requirements.txt 不存在：$req" }
if (-not (Test-Path $toy)) { Fail "toy_corpus.jsonl 不存在：$toy" }

# --- 项目目录 ---
if (-not (Test-Path $ProjectDir)) {
  New-Item -ItemType Directory -Path $ProjectDir | Out-Null
}

# --- 确保 .env （现在不用填也行，不调用 --llm 时不会用到）---
if (-not (Test-Path $dotEnv)) {
  if (Test-Path $envExample) {
    Copy-Item $envExample $dotEnv
    Write-Host "已从 .env.example 生成 .env（需要接 Qwen 时再填 OPENAI_API_KEY）"
  } else {
    @"
OPENAI_API_BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=REPLACE_WITH_YOUR_DASHSCOPE_API_KEY
OPENAI_MODEL=qwen-plus
"@ | Out-File -Encoding UTF8 $dotEnv
    Write-Host "已创建最小 .env（暂时可不改，离线检索不需要）"
  }
}

# --- toy 语料 ---
$toyDest = Join-Path $ProjectDir "toy_corpus.jsonl"
if (-not (Test-Path $toyDest)) {
  Copy-Item $toy $toyDest
  Write-Host "已拷贝 toy_corpus.jsonl 到 $ProjectDir"
}

# --- 创建/检测环境 ---
Write-Host "`n[1/3] 检查/创建 conda 环境 '$EnvName' (Python $PythonVersion) ..."
& $condaExe env list | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "conda 运行失败。" }

$envListStr = (& $condaExe env list) -join "`n"
$envExists = $envListStr -match ("^\s*" + [regex]::Escape($EnvName) + "\s")
if (-not $envExists) {
  & $condaExe create -y -n $EnvName ("python=" + $PythonVersion)
  if ($LASTEXITCODE -ne 0) { Fail "创建环境失败。" }
} else {
  Write-Host "环境 '$EnvName' 已存在。"
}

# --- 安装依赖 ---
Write-Host "`n[2/3] 安装 Python 依赖 ..."
& $condaExe run -n $EnvName python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "pip 升级失败。" }

& $condaExe run -n $EnvName python -m pip install -r $req
if ($LASTEXITCODE -ne 0) { Fail "pip 安装依赖失败。" }

# --- 冒烟测试（用临时 py 文件方式，兼容 PS 5.1）---
Write-Host "`n[3/3] 冒烟测试 ..."
$py = @"
import numpy, sklearn
print('[OK] numpy', numpy.__version__, 'scikit-learn', sklearn.__version__)
"@
$tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName() + '.py')
Set-Content -Encoding UTF8 -Path $tmp -Value $py
& $condaExe run -n $EnvName python $tmp
$code = $LASTEXITCODE
Remove-Item $tmp -Force -ErrorAction SilentlyContinue
if ($code -ne 0) { Fail "冒烟测试失败。" }

# --- Demo 运行（如果脚本存在）---
if (Test-Path $main) {
  Write-Host "`n运行 demo（离线检索，不调用 LLM）..."
  & $condaExe run -n $EnvName python $main `
    --data "$toyDest" `
    --query "How do I put the apple in the fridge?" `
    --k 6 --sim_th 0.30 --alpha 0 `
    --budget_nodes 2 --tau 1.00
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Demo 运行失败，请检查 gstar_astar_proto.py 路径或参数。"
  }
} else {
  Write-Host "未找到 $main，跳过 demo。"
}

Write-Host "`n完成。你可以直接用 conda run -n $EnvName python $main 继续测试。"
