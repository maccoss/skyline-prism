using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Reading a DIA acquisition's real isolation windows by having Skyline import them from a raw data file
/// - the only source of those windows for a "Results only" document. Uses a fake command runner, so no
/// Skyline install and no data files are needed.
/// </summary>
public class SkylineIsolationImporterTests
{
    /// <summary>
    /// Stands in for SkylineCmd: records the arguments and writes the document Skyline would have saved.
    /// </summary>
    private sealed class FakeRunner : ISkylineCommandRunner
    {
        private readonly string? _schemeXml;
        private readonly Exception? _failWith;

        public FakeRunner(string? schemeXml, Exception? failWith = null)
        {
            _schemeXml = schemeXml;
            _failWith = failWith;
        }

        public List<string[]> Invocations { get; } = new();
        public string Description => "fake runner";
        public bool SupportsParquet => false;

        public void Run(string[] args, Action<string> log, CancellationToken cancellationToken,
            TimeSpan? timeout = null)
        {
            Invocations.Add(args);
            if (_failWith is not null)
                throw _failWith;

            var newArg = args.First(a => a.StartsWith("--new=", StringComparison.Ordinal));
            var path = newArg["--new=".Length..];
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path,
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>\r\n"
                + "<srm_settings format_version=\"25.1\" software_version=\"Skyline (64-bit) 26.1\">\r\n"
                + "  <settings_summary name=\"Default\">\r\n"
                + "    <transition_settings>\r\n"
                + "      <transition_full_scan acquisition_method=\"DIA\">\r\n"
                + (_schemeXml ?? "")
                + "      </transition_full_scan>\r\n"
                + "    </transition_settings>\r\n"
                + "  </settings_summary>\r\n"
                + "</srm_settings>\r\n");
            log("Saving file...");
        }
    }

    // Real windows imported from a Thermo .raw: 3.0014 Th wide, and starting at 400.4319 rather than a
    // round number because the edges are placed in the peptide "forbidden zones". Nothing about these
    // boundaries is guessable from a uniform grid, which is the whole reason this import exists.
    private const string ForbiddenZoneScheme = """
        <isolation_scheme name="2026-03-OMAR-FLASH-Colon-691-E5-047">
          <isolation_window start="400.431890003052" end="403.433289996948" />
          <isolation_window start="403.43330995174404" end="406.434610048256" />
          <isolation_window start="406.43460000305197" end="409.435999996948" />
        </isolation_scheme>
        """;

    private static string TempDataFile()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_iso_probe_" + Guid.NewGuid().ToString("N") + ".raw");
        File.WriteAllText(path, "not really a raw file - only its existence is checked");
        return path;
    }

    [Fact]
    public void ImportFromDataFile_ReadsTheWindowsSkylineWrites()
    {
        var raw = TempDataFile();
        var runner = new FakeRunner(ForbiddenZoneScheme);
        try
        {
            var scheme = SkylineIsolationImporter.ImportFromDataFile(raw, runner, _ => { });

            Assert.NotNull(scheme);
            Assert.Equal(3, scheme!.Windows.Count);
            Assert.Equal(400.431890003052, scheme.MzLow, 9);   // NOT 400 - the edge is in a forbidden zone
            Assert.Equal(3.0014, scheme.Windows[0].Width, 4);
            // Named after the data file, so the UI can show where the windows came from.
            Assert.Equal(Path.GetFileNameWithoutExtension(raw), scheme.Name);
        }
        finally
        {
            File.Delete(raw);
        }
    }

    [Fact]
    public void ImportFromDataFile_NeverTouchesTheUsersDocument()
    {
        // The whole safety argument: the import runs against a throwaway --new document, never --in.
        var raw = TempDataFile();
        var runner = new FakeRunner(ForbiddenZoneScheme);
        try
        {
            SkylineIsolationImporter.ImportFromDataFile(raw, runner, _ => { });

            var args = Assert.Single(runner.Invocations);
            Assert.Contains(args, a => a.StartsWith("--new=", StringComparison.Ordinal));
            Assert.Contains("--overwrite", args);
            Assert.Contains("--save", args);
            Assert.Contains("--full-scan-acquisition-method=DIA", args);
            Assert.Contains("--full-scan-isolation-scheme=" + raw, args);
            // No --in / --out / --save-as: nothing points at a document the user owns.
            Assert.DoesNotContain(args, a => a.StartsWith("--in=", StringComparison.Ordinal));
            Assert.DoesNotContain(args, a => a.StartsWith("--open=", StringComparison.Ordinal));
            Assert.DoesNotContain(args, a => a.StartsWith("--out=", StringComparison.Ordinal));
            Assert.DoesNotContain(args, a => a.StartsWith("--save-as=", StringComparison.Ordinal));
            // The probe document is cleaned up.
            var newArg = args.First(a => a.StartsWith("--new=", StringComparison.Ordinal));
            Assert.False(Directory.Exists(Path.GetDirectoryName(newArg["--new=".Length..])));
        }
        finally
        {
            File.Delete(raw);
        }
    }

    [Fact]
    public void ImportFromDataFile_ReturnsNullWhenTheDataFileIsGone()
    {
        var runner = new FakeRunner(ForbiddenZoneScheme);
        Assert.Null(SkylineIsolationImporter.ImportFromDataFile(
            @"Z:\archived\away\run.raw", runner, _ => { }));
        Assert.Empty(runner.Invocations); // no point launching Skyline for a file that is not there
        Assert.Null(SkylineIsolationImporter.ImportFromDataFile(null, runner, _ => { }));
    }

    [Fact]
    public void ImportFromDataFile_ReturnsNullWhenSkylineFailsOrFindsNoWindows()
    {
        var raw = TempDataFile();
        try
        {
            // A non-DIA acquisition: the probe document comes back with no isolation scheme.
            Assert.Null(SkylineIsolationImporter.ImportFromDataFile(
                raw, new FakeRunner(schemeXml: null), _ => { }));
            // Skyline reported an error: best-effort, so the run carries on without windows.
            Assert.Null(SkylineIsolationImporter.ImportFromDataFile(
                raw, new FakeRunner(null, new InvalidOperationException("Error: bad file")), _ => { }));
        }
        finally
        {
            File.Delete(raw);
        }
    }

    [Fact]
    public void ResolveDataFile_FallsBackToTheDocumentFolderWhenPathsGoStale()
    {
        // Real case: the .sky recorded Y:\...\run.raw, but the data had been moved next to the document.
        var dir = Path.Combine(Path.GetTempPath(), "prism_iso_dir_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var document = Path.Combine(dir, "cohort.sky");
            File.WriteAllText(document, "<srm_settings />");
            var moved = Path.Combine(dir, "run.raw");
            File.WriteAllText(moved, "data");

            var resolved = SkylineIsolationImporter.ResolveDataFile(
                new[] { @"Y:\gone\run.raw" }, document);
            Assert.Equal(moved, resolved);

            // A path that still resolves is used as-is.
            Assert.Equal(moved, SkylineIsolationImporter.ResolveDataFile(new[] { moved }, document));
            // Nothing reachable anywhere -> null, and the caller asks the user instead.
            Assert.Null(SkylineIsolationImporter.ResolveDataFile(new[] { @"Y:\gone\run.raw" }, null));
            Assert.Null(SkylineIsolationImporter.ResolveDataFile(Array.Empty<string>(), document));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ResolveDataFile_StripsSkylineSampleQualifiers()
    {
        // Multi-sample WIFF paths are recorded as "file.wiff|sample name|1".
        var dir = Path.Combine(Path.GetTempPath(), "prism_iso_wiff_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var wiff = Path.Combine(dir, "batch.wiff");
            File.WriteAllText(wiff, "data");
            Assert.Equal(wiff, SkylineIsolationImporter.ResolveDataFile(
                new[] { wiff + "|sample 3|2" }, null));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
