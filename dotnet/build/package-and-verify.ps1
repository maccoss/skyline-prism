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

Write-Host '== 1/3  Full test suite ==' -ForegroundColor Cyan
dotnet test (Join-Path $root 'SkylinePrism.sln') -c $Configuration
if ($LASTEXITCODE -ne 0) { Write-Host 'ABORT: tests failed.' -ForegroundColor Red; exit 1 }

Write-Host '== 2/3  Package SkylinePrism.zip ==' -ForegroundColor Cyan
dotnet msbuild (Join-Path $root 'build\package.proj') "/p:Configuration=$Configuration"
if ($LASTEXITCODE -ne 0) { Write-Host 'ABORT: packaging failed.' -ForegroundColor Red; exit 1 }

Write-Host '== 3/3  Launch smoke test (extract zip + run the exe) ==' -ForegroundColor Cyan
pwsh -NoProfile -File (Join-Path $PSScriptRoot 'verify-tool.ps1')
if ($LASTEXITCODE -ne 0) { Write-Host 'ABORT: the packaged tool failed to launch.' -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host 'READY: SkylinePrism.zip is tested, packaged, and launch-verified.' -ForegroundColor Green
