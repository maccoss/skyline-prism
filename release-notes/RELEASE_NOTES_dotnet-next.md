# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **A `.sky.zip` no longer extracts onto the share it came from, and no longer four at a time.**
  Reported on a real Panorama cohort as "it starts 4 files and never seems to fully uncompress", and
  measured while it ran: four `.skyd` files growing at 3.7 MB/s each, ~15 MB/s together, 4.2 of 12 GB
  after fifteen minutes. Not hung - about an hour for four archives, and hours for twelve.

  dotnet-v26.24.0 extracted beside the archive, which for a Panorama download folder means reading
  ~12 GB and writing ~17 GB back over the same SMB link, and the per-destination lock let the export
  loop run four of them at once. An extraction is one I/O path, not CPU work: concurrency there buys
  nothing, costs seeks, and makes every archive finish at the end instead of one finishing early.

  An archive that is not on a local disk now extracts to the output directory when that is local, else
  the temp directory, with a log line saying where and why; `PRISM_EXTRACT_DIR` still overrides
  everything. One archive is extracted at a time, and the reuse check stays outside that gate so a
  cohort whose archives are already extracted never queues. Confirmed on the same share: **4.4 GB at
  216 MB/s**, against ~15 MB/s before.

  Extraction also reports progress now - percentage, rate and an estimate every 20 seconds, and the
  elapsed time and rate at the end. `ExtractToFile` is silent, and with 12 GB entries that silence is
  what "never seems to finish" actually looks like. And the out-of-space message named the output
  directory, which cannot help for the case that reaches it; it now names `PRISM_EXTRACT_DIR` and the
  roomiest local drive.

## Performance

## Breaking Changes
