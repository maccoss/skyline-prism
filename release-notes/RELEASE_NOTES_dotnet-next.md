# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The document's digestion enzyme is read correctly again.** Every live Skyline document logged
  "Document enzyme '...' has no PRISM equivalent; using the configured default enzyme" - including
  plain Trypsin, which PRISM has always supported. Skyline returns the enzyme over JSON-RPC as
  PascalCase `<Enzyme .../>`, while the same enzyme inside a saved `.sky` file is lowercase
  `<enzyme .../>`, and the parser matched the element name case-sensitively; only the file path ever
  worked. Element and attribute names are now matched case-insensitively. The message also
  distinguishes an unreadable definition from a rule that genuinely has no PRISM equivalent, and
  prints the cut/no_cut/sense it read. Results were not affected: the enzyme only feeds the terminus
  check in FASTA-based parsimony, and the Skyline tool does not set a FASTA path - but a `Trypsin/P`
  document driving the CLI with a FASTA would have silently fallen back to `trypsin`.

## Performance

## Breaking Changes
