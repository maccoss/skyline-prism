# Skyline-PRISM (C#) dotnet-v26.22.0 Release Notes

**This release requires the .NET 10 runtime. Install it before upgrading - the tool will not start
otherwise.** .NET 8's support window closes on 10 November 2026; .NET 10 is supported to November 2028.
The move also carries a security fix in the SQLite library behind the `.blib` reader.




## Breaking Changes

- **PRISM now targets .NET 10, so you must install the .NET 10 runtime before this version will start.**
  The Skyline tool needs the .NET 10 **Desktop** Runtime; the `prism` CLI needs the base .NET 10 Runtime.
  The download links in the README are updated.

  The reason is support, not features: .NET 8's LTS window closes on 10 November 2026, while .NET 10 is
  supported to November 2028. The artifacts stay framework-dependent, which is what keeps them at
  20-45 MB rather than bundling a runtime per platform.

  **Check your OS first if it is an old one.** .NET 9 dropped Windows 8.1 and Server 2012/2012 R2, and
  .NET 10 raises the minimum macOS and Linux glibc versions - so on an older machine the answer is an OS
  upgrade rather than a runtime install, and the installer will simply refuse. The supported-platforms
  list is at <https://github.com/dotnet/core/blob/main/release-notes/10.0/supported-os.md>.

  Alongside the runtime bump, **Microsoft.Data.Sqlite moves 8.0.10 -> 10.0.0 and
  SQLitePCLRaw.lib.e_sqlite3 is pinned to 2.1.13**, which clears GHSA-2m69-gcr7-jv3q - a high-severity
  advisory against the bundled SQLite native library used to read `.blib` spectral libraries. The
  advisory applied to every previous release; it went unnoticed because the older SDK did not audit
  transitive dependencies. Bumping Microsoft.Data.Sqlite alone was not enough (it resolves 2.1.11, same
  advisory), hence the direct pin. Dependabot now watches NuGet so the next one does not sit as long.

  Verified beyond the test suite, because a runtime change moves GC and marshalling behavior and that is
  where DuckDB.NET has broken here before: a 47M-row, 93-sample cohort was run end to end through the
  streaming merge and the Stage 2 reader with no failure.
