param(
    [string]$param1,
    [string]$param2,
    [string]$param3
)

$arguments = @()

if ($param1) { $arguments += $param1 }
if ($param2) { $arguments += $param2 }
if ($param3) { $arguments += $param3 }

Write-Host "Running ParallelJacobian.exe without arguments..."
if ($arguments.Count -eq 0) {
    Start-Process -FilePath ".\ParallelJacobian.exe" -Wait
} else {
    Start-Process -FilePath ".\ParallelJacobian.exe" -ArgumentList $arguments -Wait
}

Write-Host "Running draw.py..."
python draw.py
