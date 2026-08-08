<#
  Smoke-test the packaged Skyline tool BEFORE shipping/installing it.

  Extracts SkylinePrism.zip to a clean directory (a fresh-install simulation) and launches
  SkylinePrism.exe with a dummy connection argument. The MainWindow - and thus ScottPlot/SkiaSharp -
  loads during startup, so a missing or broken dependency surfaces as an assembly/XAML load error in
  prism-tool.log. A failed connection from the dummy arg is expected and ignored; only binary/
  dependency load failures fail the check. This catches exactly the class of failure that a
  "the zip is complete" file check cannot (e.g. a XAML/SkiaSharp load regression).

  Exit code 0 = launched and loaded the UI cleanly; 1 = load failure (or the exe is missing).

  Usage:
    pwsh dotnet/build/verify-tool.ps1
    pwsh dotnet/build/verify-tool.ps1 -Zip path\to\SkylinePrism.zip -WaitSeconds 8
#>
param(
    [string]$Zip = (Join-Path $PSScriptRoot '..\publish\SkylinePrism.zip'),
    # Upper bound, not a dwell time: the run below polls the log and stops as soon as the tool has
    # either loaded its window or crashed, so the normal case takes about a second. The bound only has
    # to cover a cold start on the slowest machine that runs this - a CI runner, not this laptop.
    [int]$WaitSeconds = 40
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Zip)) { Write-Host "VERIFY FAILED: zip not found: $Zip"; exit 1 }

$log = Join-Path $env:LOCALAPPDATA 'SkylinePrism\prism-tool.log'
$startLen = if (Test-Path $log) { (Get-Item $log).Length } else { 0 }

$dir = Join-Path $env:TEMP 'prism-tool-verify'
Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $dir)

$exe = Join-Path $dir 'SkylinePrism.exe'
if (-not (Test-Path $exe)) { Write-Host 'VERIFY FAILED: SkylinePrism.exe not found in the zip'; exit 1 }

# Only what THIS launch appended to the shared log. Opened share-write, because the tool still has it
# open; the tool logs with a lock and appends per line, so a read mid-run is safe.
function Get-LogTail {
    if (-not (Test-Path $log)) { return '' }
    $fs = [System.IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
    try {
        [void]$fs.Seek($startLen, 'Begin')
        $sr = New-Object System.IO.StreamReader($fs)
        try { return $sr.ReadToEnd() } finally { $sr.Close() }
    } finally { $fs.Dispose() }
}

Write-Host "Launching (extracted) $exe ..."
$p = Start-Process -FilePath $exe -ArgumentList 'verify-smoke-no-connection' -PassThru

# Poll for a verdict rather than sleeping a fixed period. A fixed wait has to be long enough for the
# slowest machine, which makes it slow everywhere and STILL flaky on a cold CI runner; polling is both
# faster here and safe there. Either marker below is conclusive, so there is nothing to gain by waiting.
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
    if ((Get-LogTail) -match 'UNHANDLED \(|MainWindow loaded') { break }
    if ($p.HasExited) { break }
    Start-Sleep -Milliseconds 250
}
if (-not $p.HasExited) { $p.Kill(); [void]$p.WaitForExit(5000) }

$tail = Get-LogTail   # final read, after the process has gone and flushed
Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue

# ANY exception that escaped to App's global handlers fails the check. This deliberately replaces an
# allowlist of five load-failure strings, which let a startup crash through: an NRE thrown out of
# MainWindow's constructor is logged as "UNHANDLED (UI thread): ...TargetInvocationException", matches
# none of those five, and - because App sets e.Handled to keep the error dialog readable - leaves the
# process alive, so the wait below looked healthy too. A failed Skyline connection (expected with the
# dummy arg) is caught by the driver and never reaches these handlers, so it still will not fail here.
if ($tail -match 'UNHANDLED \(') {
    Write-Host 'VERIFY FAILED: an exception escaped to the tool''s global handler on startup:'
    Write-Host '----------------------------------------------------------------------'
    Write-Host $tail
    exit 1
}
if ($tail -notmatch 'Skyline-PRISM tool started') {
    Write-Host 'VERIFY FAILED: the tool did not start (no startup entry in the log).'
    Write-Host $tail
    exit 1
}
# The startup line above is written in OnStartup, BEFORE base.OnStartup builds MainWindow - so on its
# own it only proves the process ran, not that the UI came up. MainWindow logs this once it is fully
# constructed and shown; requiring it is what makes this a launch verification rather than a smoke test.
if ($tail -notmatch 'MainWindow loaded') {
    Write-Host 'VERIFY FAILED: the process started but the main window never finished loading.'
    Write-Host $tail
    exit 1
}
Write-Host 'VERIFY PASSED: SkylinePrism.exe launched from the packaged zip, built its main window, and logged no unhandled exception.'
exit 0
