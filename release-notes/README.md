# Release Notes

This directory contains per-version release notes for Skyline-PRISM.

## Versioning Scheme

Skyline-PRISM uses a `YY.feature.patch` versioning convention:

- **YY**: Two-digit year (e.g., `26` for 2026)
- **feature**: Incremented for each release containing new features
- **patch**: Incremented for bug-fix-only releases within the same feature version

Examples: `26.1.0` (first feature release of 2026), `26.1.1` (patch), `26.2.0` (second feature release).

The package version in `pyproject.toml` is updated only at release time, not during development.

## File Format

Each release gets one file: `RELEASE_NOTES_v{version}.md`. During development, the unreleased draft lives in `RELEASE_NOTES_next.md` and gets renamed at release time.

```text
release-notes/
  README.md                      # this file
  RELEASE_NOTES_next.md          # working draft for the next release
  RELEASE_NOTES_v26.3.3.md
  RELEASE_NOTES_v26.4.0.md
```

## Writing Release Notes

### During Development

Maintain `RELEASE_NOTES_next.md` as a working draft for the next planned version. Append entries as features and fixes land on the development branch. The file is unversioned until the release is finalized so the target version can change (e.g., a planned patch release becomes a feature release once new functionality is added).

### Content Structure

Each release notes file should use this structure:

```markdown
# Skyline-PRISM v{version} Release Notes

One-sentence summary of the release.

## New Features

- Feature descriptions grouped by area (e.g., Rollup, Batch Correction, QC)
- Focus on what changed from the user's perspective, not implementation details

## Bug Fixes

- Description of the bug and its impact
- What was fixed

## Performance

- Performance improvements with context (e.g., "Reduced memory from 35 GB to 8 GB for 100-parquet experiments")

## Breaking Changes

- Any changes that require user action (config format changes, removed options, etc.)
- Omit this section if there are no breaking changes
```

Sections can be omitted if empty. For major releases with many changes, subsections within each category are fine. For patch releases, a flat list is sufficient.

> [!IMPORTANT]
> **Delete the empty headings when you rename the draft.** Both rolling drafts (`RELEASE_NOTES_next.md`
> and `RELEASE_NOTES_dotnet-next.md`) are seeded with all four headings so entries have somewhere to go
> during development — which means a renamed draft *always* arrives carrying the ones nobody filled in.
> Removing them is a step of the release, not something the draft gets right on its own. It matters most
> on the C# track, where the file is published verbatim as the GitHub Release description and empty
> headings are visible to everyone reading the Releases page.

### Style

- Write in past tense ("Added", "Fixed", "Removed")
- Lead with user impact, not implementation details
- Include specific numbers where relevant (e.g., memory reduction, sample counts, file sizes)
- Reference config options by their CLI flag or YAML key name
- Reference modified files with paths so reviewers can locate the change

## Two release tracks: Python and C# (.NET)

Skyline-PRISM ships two implementations side by side, released independently:

- **Python** — the `skyline-prism` PyPI package. Notes: `RELEASE_NOTES_v{version}.md`
  (draft `RELEASE_NOTES_next.md`). Version source: `pyproject.toml`. Tag: `v{version}`.
- **C# (.NET)** — the `prism` CLI + Windows Skyline external tool, published to GitHub Releases.
  Notes: `RELEASE_NOTES_dotnet-v{version}.md` (draft `RELEASE_NOTES_dotnet-next.md`). Version source:
  `dotnet/Directory.Build.props` **and** `dotnet/src/SkylinePrism.App/tool-inf/info.properties`
  (kept in lockstep). Tag: `dotnet-v{version}`.

Both tracks use the same CalVer `YY.feature.patch` scheme but keep **distinct tag namespaces and
notes filenames** so their counters never collide.

## Python Release Process

1. Finalize `RELEASE_NOTES_next.md` on the development branch
2. Rename it: `git mv release-notes/RELEASE_NOTES_next.md release-notes/RELEASE_NOTES_v{version}.md`
3. Update the title heading inside the file to match the version, and **delete every section heading
   with no entries under it** — the draft is seeded with all four
4. Create a fresh empty `RELEASE_NOTES_next.md` for the following release
5. Update `version` in `pyproject.toml` **and** `__version__` in `skyline_prism/__init__.py` to match
6. Commit the version bump and renames
7. Merge to `main`
8. Tag: `git tag v{version}`
9. Push: `git push origin main --tags` — then publish a GitHub Release, which triggers PyPI upload.

## C# (.NET) Release Process

1. Finalize `RELEASE_NOTES_dotnet-next.md`; rename to `RELEASE_NOTES_dotnet-v{version}.md`, update its
   heading, and **delete every section heading with no entries under it** (the draft is seeded with all
   four, and this file is published verbatim as the Release body); create a fresh empty
   `RELEASE_NOTES_dotnet-next.md`
2. Bump the version in **both** `dotnet/Directory.Build.props` (`<Version>`) and
   `dotnet/src/SkylinePrism.App/tool-inf/info.properties` (`Version =`) to `{version}` — they must
   match, and `dotnet-release.yml` fails the release if either differs from the tag
3. Commit and merge to `main`
4. Tag: `git tag dotnet-v{version}`
5. Push the tag: `git push origin dotnet-v{version}` — **pushing the tag both builds the artifacts
   and creates the GitHub Release** (via `dotnet-release.yml`). Do **not** hand-create the GitHub
   Release. There is no PyPI step for the C# track.

> [!IMPORTANT]
> **This file becomes the GitHub Release description.** `dotnet-release.yml` publishes
> `release-notes/RELEASE_NOTES_${tag}.md` verbatim as the Release body — so write it for the people
> reading the Releases page, not just for the repo. Step 1's rename therefore has to happen **before**
> tagging: the workflow resolves the path from the tag and fails with an explicit message if the file is
> absent, after the artifacts have already built.
>
> The Python track is the other way round — you create the Release by hand (that is what triggers the
> PyPI upload), so paste the notes in yourself.
>
> To fix an existing Release:
> `gh release edit <tag> --notes-file release-notes/RELEASE_NOTES_<tag>.md`
