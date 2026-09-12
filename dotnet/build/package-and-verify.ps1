<#
  Ship gate for the Skyline external tool: test -> package -> launch-verify, in that order.
  Use this instead of calling package.proj directly so the tool is always exercised before it is
  shipped or installed. Any step failing aborts with a non-zero exit code and no zip is declared ready.

  Usage:
    pwsh dotnet/build/package-and-verify.ps1
    pwsh dotnet/build/package-and-verify.ps1 -Configuration Debug
#>
param([string]$Configuration = 'Release')
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')

Write-Host '== 1/4  Full test suite ==' -ForegroundColor Cyan
dotnet test (Join-Path $root 'SkylinePrism.sln') -c $Configuration
if ($LASTEXITCODE -ne 0) { Write-Host 'ABORT: tests failed.' -ForegroundColor Red; exit 1 }

Write-Host '== 2/4  Package SkylinePrism.zip ==' -ForegroundColor Cyan
dotnet msbuild (Join-Path $root 'build\package.proj') "/p:Configuration=$Configuration"
if ($LASTEXITCODE -ne 0) { Write-Host 'ABORT: packaging failed.' -ForegroundColor Red; exit 1 }

# The reader is what makes acquired ion counts measurable at all, and every published artifact
# carries one - the tool zip here, and each `prism` CLI via dotnet-release.yml. package.proj forces
# PrismWithPwiz=true, so a zip without it means the pwiz checkout was missing or the reference
# stopped resolving - and the symptom would be silent: the tool launches, passes the smoke test
# below, and the Ion accounting pane simply never appears, because it hides itself when nothing has
# been measured.
Write-Host '== 3/4  Check the packaged tool carries the instrument-file reader ==' -ForegroundColor Cyan
$zip = Join-Path $root 'publish\SkylinePrism.zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
    $names = $archive.Entries | ForEach-Object { Split-Path $_.FullName -Leaf }
} finally {
    $archive.Dispose()
}
$missing = @('SkylinePrism.Pwiz.dll', 'Pwiz.Data.MsData.dll') | Where-Object { $names -notcontains $_ }
if ($missing) {
    Write-Host ("ABORT: the packaged zip has no instrument-file reader (missing: " +
        ($missing -join ', ') + "). Acquired ion counts would be unmeasurable and the Ion " +
        "accounting pane would never appear. Check that the pwiz-sharp checkout is present " +
        "and that Pwiz.props resolves it.") -ForegroundColor Red
    exit 1
}
Write-Host ("  reader present: " + (($names | Where-Object { $_ -like 'Pwiz*' -or $_ -eq 'SkylinePrism.Pwiz.dll' }).Count) + " pwiz assemblies in the zip.")

Write-Host '== 4/4  Launch smoke test (extract zip + run the exe) ==' -ForegroundColor Cyan
pwsh -NoProfile -File (Join-Path $PSScriptRoot 'verify-tool.ps1')
if ($LASTEXITCODE -ne 0) { Write-Host 'ABORT: the packaged tool failed to launch.' -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host 'READY: SkylinePrism.zip is tested, packaged, reader-checked, and launch-verified.' -ForegroundColor Green
