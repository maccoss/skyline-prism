# Skyline-PRISM (C#) dotnet-v26.7.2 Release Notes

Small patch on top of 26.7.1: the run log is readable again when several Skyline documents export at
once, and GitHub Releases now actually carry these notes.

## Bug Fixes

- **Log output from concurrent exports is now attributable.** With several documents exporting at once,
  the lines forwarded from Skyline's own console ("Opening file...", "Success! Imported Reports", "2%",
  "3%") carried nothing identifying the document, so the log read as a stream of duplicated pairs. Every
  line produced while preparing an input is now tagged with its batch label:

  ```text
  [Plate1] Exporting the PRISM transition report via Skyline-daily (headless application)...
  [Plate1]     Opening file...
  [Plate2]     Opening file...
  [Plate1]     2%
  [Plate2]     2%
  ```

  The tag is applied once at the boundary, so it covers documents open in Skyline, closed ones, already-
  exported reports, and the Skyline process output alike - the live-RPC path had no labelling at all
  before. It sits in front of the indentation so the structure still reads, and blank lines stay untagged
  so they still separate sections.

- **GitHub Releases were published with an empty description.** The release workflow never checked out
  the repository, so the notes file it should have used was not on disk and every release on the C# track
  was created with no body. The workflow now checks out the tag, resolves
  `release-notes/RELEASE_NOTES_<tag>.md`, and publishes it as the release description - failing with an
  explicit message if that file is missing, rather than silently publishing an empty release. The earlier
  releases (v26.5.0 through v26.7.1) have been backfilled with their notes.
