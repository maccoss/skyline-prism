# Release Notes

This directory contains per-version release notes for Skyline-PRISM.

## Versioning Scheme

Skyline-PRISM uses a `YY.feature.patch` versioning convention:

- **YY**: Two-digit year (e.g., `26` for 2026)
- **feature**: Incremented for each release containing new features
- **patch**: Incremented for bug-fix-only releases within the same feature version

Examples: `26.1.0` (first feature release of 2026), `26.1.1` (patch), `26.2.0` (second feature release).

The version is bumped only at release time, not during development.

## File Format

Each release gets one file: `RELEASE_NOTES_dotnet-v{version}.md`. During development, the unreleased
draft lives in `RELEASE_NOTES_dotnet-next.md` and gets renamed at release time.

```text
release-notes/
  README.md                          # this file
  RELEASE_NOTES_dotnet-next.md       # working draft for the next release
  RELEASE_NOTES_dotnet-v26.15.0.md
  RELEASE_NOTES_v26.4.4.md           # ...and the frozen notes of the retired Python track
```

> [!NOTE]
> The `RELEASE_NOTES_v*.md` files (no `dotnet-` prefix) belong to the **retired Python package**, whose
> last release was `v26.4.4`. They are kept because those tags and PyPI versions are still downloadable.
> Do not add new ones.

## Writing Release Notes

### During Development

Maintain `RELEASE_NOTES_dotnet-next.md` as a working draft for the next planned version. Append entries as features and fixes land on the development branch. The file is unversioned until the release is finalized so the target version can change (e.g., a planned patch release becomes a feature release once new functionality is added).

### Content Structure

Each release notes file should use this structure:

```markdown
# Skyline-PRISM (C#) dotnet-v{version} Release Notes

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
> **Delete the empty headings when you rename the draft.** The rolling draft
> (`RELEASE_NOTES_dotnet-next.md`) is seeded with all four headings so entries have somewhere to go
> during development — which means a renamed draft *always* arrives carrying the ones nobody filled in.
> Removing them is a step of the release, not something the draft gets right on its own, and it is
> visible to everyone: the file is published verbatim as the GitHub Release description.

### Style

- Write in past tense ("Added", "Fixed", "Removed")
- Lead with user impact, not implementation details
- Include specific numbers where relevant (e.g., memory reduction, sample counts, file sizes)
- Reference config options by their CLI flag or YAML key name
- Reference modified files with paths so reviewers can locate the change

## The release track

Skyline-PRISM ships the **C# (.NET)** tools — the `prism` CLI + the Windows Skyline external tool —
published to GitHub Releases. Notes: `RELEASE_NOTES_dotnet-v{version}.md` (draft
`RELEASE_NOTES_dotnet-next.md`). Version sources: `dotnet/Directory.Build.props` **and**
`dotnet/src/SkylinePrism.App/tool-inf/info.properties` (kept in lockstep). Tag: `dotnet-v{version}`.

The `dotnet-` prefix is kept rather than dropped: the retired Python track used the bare `v{version}`
namespace, and renaming would break existing tags, release URLs, and the workflow's tag → notes-file
mapping.

## Release Process

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
   Release.

> [!IMPORTANT]
> **This file becomes the GitHub Release description.** `dotnet-release.yml` publishes
> `release-notes/RELEASE_NOTES_${tag}.md` verbatim as the Release body — so write it for the people
> reading the Releases page, not just for the repo. Step 1's rename therefore has to happen **before**
> tagging: the workflow resolves the path from the tag and fails with an explicit message if the file is
> absent, after the artifacts have already built.
>
> To fix an existing Release:
> `gh release edit <tag> --notes-file release-notes/RELEASE_NOTES_<tag>.md`
