param(
    [ValidateSet('smoke','full','index-train','train','index-test','predict','validate')]
    [string]$Mode = 'smoke',
    [string]$Dataset = '',
    [int]$SampleSize = 20000,
    [int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path (Split-Path $projectRoot -Parent) -Parent
if (-not $Dataset) {
    $Dataset = Join-Path $workspaceRoot '6ab10eb3b23ba_student_resource\student_resource\dataset'
}
if (-not (Test-Path -LiteralPath $Dataset)) { throw "Dataset not found: $Dataset" }
$runtimeRoot = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $runtimeRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & python -m venv $runtimeRoot
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
$requirements = Join-Path $projectRoot 'requirements.txt'
$requirementHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
$installStamp = Join-Path $runtimeRoot 'requirements.sha256'
if (-not (Test-Path -LiteralPath $installStamp) -or (Get-Content -LiteralPath $installStamp -Raw).Trim() -ne $requirementHash) {
    & $pythonPath -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Set-Content -LiteralPath $installStamp -Value $requirementHash
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $projectRoot 'src'
try {
    $work = Join-Path $workspaceRoot 'runs\local\work'
    $output = Join-Path $workspaceRoot 'output'
    if ($Mode -eq 'smoke') {
        $smokeDataset = Join-Path $workspaceRoot 'runs\smoke\dataset'
        $work = Join-Path $workspaceRoot 'runs\smoke\work'
        $output = Join-Path $workspaceRoot 'runs\smoke\output'
        if (-not (Test-Path -LiteralPath (Join-Path $smokeDataset 'SMOKE_ONLY.json'))) {
            & $pythonPath -m er make-smoke --dataset $Dataset --destination $smokeDataset
            if ($LASTEXITCODE -ne 0) { throw 'Smoke dataset creation failed.' }
        }
        & $pythonPath -m er run --dataset $smokeDataset --work $work --output $output --sample-size 1000 --threads $Threads
    } elseif ($Mode -eq 'full') {
        & $pythonPath -m er run --dataset $Dataset --work $work --output $output --sample-size $SampleSize --threads $Threads
    } elseif ($Mode -eq 'index-train') {
        & $pythonPath -m er index --dataset $Dataset --work $work --split train
    } elseif ($Mode -eq 'index-test') {
        & $pythonPath -m er index --dataset $Dataset --work $work --split test
    } elseif ($Mode -eq 'train') {
        & $pythonPath -m er train --dataset $Dataset --work $work --sample-size $SampleSize --threads $Threads
    } elseif ($Mode -eq 'predict') {
        & $pythonPath -m er predict --dataset $Dataset --work $work --output $output --threads $Threads
    } else {
        & $pythonPath -m er validate --dataset $Dataset --work $work --output $output
    }
    if ($LASTEXITCODE -ne 0) { throw "Pipeline stopped with exit code $LASTEXITCODE. See message above." }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
