# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- An output folder copied over an older copy of itself no longer has stray merged data read as real
  rows. A copy that adds and overwrites files but never deletes them keeps any `merged_data/` partition
  the newer run no longer wrote; a real cohort folder held three partitions where its merge had written
  two, putting 9.9 million duplicated rows (10,494 peptides) in front of every reader of `merged_data/` -
  the Spectrum density view, detection and ion accounting - with nothing to say so. Each merge now writes `merged_data/_manifest.json` listing its partitions and
  files, and a `merged_data/` that does not match it is not read: the error names every stray, missing
  or changed file and says how to fix it. `prism run` rebuilds such a folder, and every stage below it,
  instead of stopping. A `merged_data/` written by an earlier release is checked against the row count
  in `merged_data.cache.json` instead, which catches the same case (39,789,408 rows on disk against
  29,910,240 recorded, in the folder that was found). See `docs/output_files.md`.
- Corrected documentation that contradicted the code, with no change to any output. `docs/output_files.md`
  and `SPECIFICATION.md` listed a `Coeluting` column the bundled `Skyline-PRISM` report has never exported,
  and `docs/output_files.md` omitted the `Background` column it does export (nothing reads it).

## Performance

## Breaking Changes
