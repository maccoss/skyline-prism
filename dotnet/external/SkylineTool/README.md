# Vendored Skyline JSON-RPC client sources

These four files are copied verbatim from
`pwiz_tools/Skyline/SkylineTool/` in the
[ProteoWizard pwiz](https://github.com/ProteoWizard/pwiz) repository (Apache
License 2.0). They define the JSON-RPC interface and client used by Skyline
external tools to read and write the active document.

- `SkylineJsonToolClient.cs`
- `IJsonToolService.cs`
- `JsonToolConstants.cs`
- `JsonToolModels.cs`

The `SkylineCadenza.Core` project link-compiles them via `<Compile Include="..." Link="..." />`
entries in `SkylineCadenza.Core.csproj`. This matches the pattern used by pwiz's own
`SkylineMcpServer.csproj`.

To sync upstream changes: re-copy the four files from a recent pwiz checkout. Diff
before committing - the files are independent of pwiz's other internal APIs but the
upstream is the source of truth.
