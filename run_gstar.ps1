# run_gstar.ps1 - Quick run inside the conda env
param(
  [string]$EnvName = "gstar310",
  [string]$ProjectDir = (Get-Location).Path,
  [string]$MainPy = "gstar_astar_proto.py"
)
$condaExe = $env:CONDA_EXE
if (-not $condaExe) { $condaExe = (Get-Command conda.exe -ErrorAction SilentlyContinue)?.Source }
if (-not $condaExe) { throw "Conda not found." }

$toy = Join-Path $ProjectDir "toy_corpus.jsonl"
$main = Join-Path $ProjectDir $MainPy
& $condaExe run -n $EnvName python $main --data "$toy" --query "How do I put the apple in the fridge?" --k 6 --sim_th 0.30 --alpha 0 --budget_nodes 2 --tau 1.00
