param (
    [int]$param1,
    [int]$param2,
    [int]$param3
)

for ($i = 1; $i -le 5; $i++) {
    Write-Host "Iteration $i running..."
    .\run_sparse.ps1 $param1 $param2 $param3

    $resultsPath = Join-Path -Path ".." -ChildPath "results"
    $newResultsPath = Join-Path -Path ".." -ChildPath ("results_" + $i)

    while (!(Test-Path $resultsPath)) {
        Start-Sleep -Seconds 1
    }

    if (Test-Path $newResultsPath) {
        Write-Host "Warning: Folder $newResultsPath already exists. Deleting it."
        Remove-Item -Recurse -Force $newResultsPath
    }

    Rename-Item -Path $resultsPath -NewName ("results_" + $i)
    Write-Host "Iteration $i complete.`n"
}
