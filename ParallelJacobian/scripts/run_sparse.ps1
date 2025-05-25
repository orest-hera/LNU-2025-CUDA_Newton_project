param(
    [string]$param1,
    [string]$param2,
    [string]$param3
)

$resultsPath = "..\results"

if (Test-Path -Path $resultsPath) {
    Write-Host "Clearing existing 'results' directory..."
    Remove-Item -Path "$resultsPath\*" -Recurse -Force
} else {
    Write-Host "Creating 'results' directory..."
    New-Item -ItemType Directory -Path $resultsPath | Out-Null
}

$arguments = @()

if ($param1) { $arguments += $param1 }
if ($param2) { $arguments += $param2 }
if ($param3) { $arguments += $param3 }

Write-Host "Running ParallelJacobian.exe..."
if ($arguments.Count -eq 0) {
    Start-Process -FilePath "..\build\Debug\ParallelJacobian_SparsityStudy.exe" -Wait
} else {
    Start-Process -FilePath "..\build\Debug\ParallelJacobian_SparsityStudy.exe" -ArgumentList $arguments -Wait
}

Write-Host "Checking for pip..."
$pipVersion = & python -m pip --version 2>$null

if (!$pipVersion) {
    Write-Host "pip is not installed. Please install pip first." -ForegroundColor Red
    exit 1
}

Write-Host "Installing required Python packages from requirements.txt..."
& python -m pip install --upgrade pip
& python -m pip install -r requirements.txt

Write-Host "Running draw.py..."
python draw.py
