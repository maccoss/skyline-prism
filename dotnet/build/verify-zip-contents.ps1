<#
  What SkylinePrism.zip must contain to be a working Skyline external tool.

  Every check here is for something whose ABSENCE IS SILENT. The zip still builds, extracts and
  launches without any of it; the loss shows up later, in the Tool Store or on someone's machine,
  with nothing in a green CI run to have warned them.

  Used by the local ship gate (package-and-verify.ps1), the CI pack job, and the release workflow,
  so the three cannot disagree about what a complete package is.

  Usage:
    pwsh dotnet/build/verify-zip-contents.ps1 -Zip dotnet/publish/SkylinePrism.zip
    pwsh dotnet/build/verify-zip-contents.ps1 -Zip ... -RequireReader
#>
param(
    [Parameter(Mandatory = $true)][string]$Zip,
    # The instrument-file reader is deliberately absent from the CI pack job, which packages with
    # AllowNoPwiz to avoid cloning ~270 MB of pwiz-sharp on every push. Every PUBLISHED zip has it.
    [switch]$RequireReader
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Zip)) { Write-Host "ABORT: no zip at $Zip" -ForegroundColor Red; exit 1 }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $Zip))
try {
    $entries = @($archive.Entries | ForEach-Object {
        [pscustomobject]@{ Path = $_.FullName -replace '\\', '/'; Length = $_.Length }
    })

    $problems = @()

    # ---- the manifest Skyline reads to install the tool at all
    foreach ($required in @('tool-inf/info.properties', 'tool-inf/SkylinePrism.properties',
                            'SkylinePrism.exe')) {
        if (-not ($entries.Path -contains $required)) { $problems += "missing $required" }
    }

    # ---- the Tool Store logo
    #
    # The Store takes the IMAGE in tool-inf/ and shows it as the tool's logo; with none, the tool
    # lists with a blank. It reaches the zip through a single <None Include="..\..\..\images\..."
    # Link="tool-inf\..."> line in SkylinePrism.App.csproj (PR #111, Vagisha Sharma) - so a moved
    # image, a renamed folder or a tidied csproj removes it, and NOTHING else notices: the zip
    # builds, extracts, launches and installs exactly as before.
    $images = @($entries | Where-Object { $_.Path -match '^tool-inf/.+\.(png|jpg|jpeg|bmp|gif)$' })
    if ($images.Count -eq 0) {
        $problems += ("no image in tool-inf/ - the Skyline Tool Store shows the image there as the " +
            "tool logo, so the listing would be blank. Check the <None Include> for " +
            "images/skyline-prism-logo.png in SkylinePrism.App.csproj")
    }
    elseif ($images.Count -gt 1) {
        # Which one the Store picks is not defined, so two is not "belt and braces", it is a coin flip.
        $problems += ("tool-inf/ carries $($images.Count) images (" +
            (($images.Path) -join ', ') + ") - the Tool Store picks one and it is not defined which")
    }
    else {
        $logo = $images[0]
        # A zero-length or placeholder file passes an existence check and still shows as nothing.
        if ($logo.Length -lt 1024) {
            $problems += "$($logo.Path) is only $($logo.Length) bytes - that is not an image"
        }
        else {
            $entry = $archive.GetEntry($logo.Path)
            $stream = $entry.Open()
            try {
                $head = New-Object byte[] 8
                $read = $stream.Read($head, 0, 8)
            } finally { $stream.Dispose() }
            $png = @(0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
            if ($read -lt 8 -or (Compare-Object $head $png)) {
                $problems += "$($logo.Path) is not a PNG (bad magic bytes)"
            }
        }
    }

    # ---- the instrument-file reader, on published builds
    if ($RequireReader) {
        foreach ($dll in @('SkylinePrism.Pwiz.dll', 'Pwiz.Data.MsData.dll')) {
            if (-not ($entries.Path -contains $dll)) {
                $problems += ("missing $dll - acquired ion counts would be unmeasurable and the " +
                    "Ion accounting pane would never appear. Check that the pwiz-sharp checkout " +
                    "is present and that Pwiz.props resolves it")
            }
        }
    }

    if ($problems.Count -gt 0) {
        Write-Host "ABORT: $Zip is not a complete tool package:" -ForegroundColor Red
        foreach ($p in $problems) { Write-Host "  - $p" -ForegroundColor Red }
        exit 1
    }

    $logoName = if ($images.Count -eq 1) { Split-Path $images[0].Path -Leaf } else { 'none' }
    $readers = @($entries | Where-Object { $_.Path -like 'Pwiz*' -or $_.Path -eq 'SkylinePrism.Pwiz.dll' }).Count
    Write-Host ("  package contents OK: manifest, logo ($logoName, " +
        "$([math]::Round($images[0].Length / 1KB)) KB), $readers pwiz assemblies.")
}
finally {
    $archive.Dispose()
}
