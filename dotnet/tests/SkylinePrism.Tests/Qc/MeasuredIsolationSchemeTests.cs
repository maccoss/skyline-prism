using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Keeping the acquisition's real isolation windows after the instrument files are gone.
///
/// <para>The windows live in exactly two places once a run is over: the data files, and what PRISM
/// wrote down beside the outputs. A DIA analysis document stores none of its own - Skyline reads them
/// at import and records <c>isolation_scheme name="Results only"</c> - and the data files are the first
/// thing to be moved off a share when an analysis is finished. At that point nothing can say what the
/// data was acquired with, and the Spectrum density map silently falls back to a built-in layout that
/// looks like a plausible DIA cycle and is not this one. These tests pin the writing-down.</para>
/// </summary>
public class MeasuredIsolationSchemeTests
{
    private static IsolationScheme Measured() => new("Imported from run01", new[]
    {
        new IsolationWindow(400.0, 403.0014),
        new IsolationWindow(403.0014, 406.0028),
        new IsolationWindow(406.0028, 409.0042),
    });

    private static IsolationScheme Declared() => new("SWATH (25 m/z)", new[]
    {
        new IsolationWindow(400, 425, 0.5),
        new IsolationWindow(425, 450, 0.5),
    });

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_measured_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// THE case: the windows and the file they were read from survive a round trip to disk, so a
    /// reopened output directory knows the acquisition without the acquisition being reachable.
    /// </summary>
    [Fact]
    public void AMeasuredSchemeSurvivesTheFileItWasReadFrom()
    {
        var dir = TempDir();
        try
        {
            var when = new DateTime(2026, 9, 12, 4, 11, 0, DateTimeKind.Utc);
            var catalog = new IsolationSchemeCatalog();
            catalog.AddMeasuredScheme(Measured(), @"R:\cohort\run01.raw", when);

            var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
            catalog.Save(path);
            var loaded = IsolationSchemeCatalog.Load(path);

            Assert.NotNull(loaded);
            var record = Assert.Single(loaded!.Measured);
            Assert.Equal(@"R:\cohort\run01.raw", record.DataFile);
            Assert.Contains("2026-09-12T04:11:00", record.MeasuredUtc);
            Assert.Equal(3, record.Scheme.Windows.Count);
            Assert.Equal(409.0042, record.Scheme.MzHigh, 6);
            Assert.True(loaded.IsMeasured(record.Scheme));

            // Offered to the picker exactly once. It is written twice on purpose - inside <measured>
            // and in the library dump, so an older PRISM reading the file still offers it - and the
            // load has to collapse that back to one entry or the list shows a duplicate.
            Assert.Single(loaded.UsableSchemes);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// Re-reading the same acquisition replaces the record rather than stacking one per run - the
    /// directory would otherwise grow an entry every time the tab was opened.
    /// </summary>
    [Fact]
    public void ReReadingTheSameLayoutReplacesTheRecord()
    {
        var catalog = new IsolationSchemeCatalog();
        catalog.AddMeasuredScheme(Measured(), @"R:\cohort\run01.raw");
        catalog.AddMeasuredScheme(Measured(), @"R:\cohort\run02.raw");

        var record = Assert.Single(catalog.Measured);
        Assert.Equal(@"R:\cohort\run02.raw", record.DataFile);
        Assert.Single(catalog.UsableSchemes);
    }

    /// <summary>
    /// Re-reading the SAME acquisition from a different plate's folder names the scheme after that
    /// folder's file, so one layout acquires a second name. The library used to be guarded on the
    /// layout AND the name, which left the first name orphaned - and since a measured scheme is
    /// recognized by its layout alone, BOTH then counted as measured. The resolver's "the measured
    /// one wins" tie-break saw two and gave up, so the directory could no longer resolve its own
    /// cached scheme without being told which by name.
    /// </summary>
    [Fact]
    public void ReReadingOneLayoutUnderASecondNameDoesNotOrphanTheFirst()
    {
        var dir = TempDir();
        try
        {
            var windows = Measured().Windows;
            var catalog = new IsolationSchemeCatalog();
            catalog.AddMeasuredScheme(
                new IsolationScheme("Imported from plateA-run01", windows), @"R:\plateA\run01.raw");
            catalog.AddMeasuredScheme(
                new IsolationScheme("Imported from plateB-run50", windows), @"R:\plateB\run50.raw");

            Assert.Single(catalog.Measured);
            Assert.Single(catalog.UsableSchemes);

            catalog.Save(Path.Combine(dir, IsolationSchemeCatalog.FileName));
            var log = new System.Collections.Generic.List<string>();
            var resolved = IsolationSchemeResolver.Resolve(dir, rawDir: null, log.Add);

            Assert.NotNull(resolved);
            Assert.Equal("Imported from plateB-run50", resolved!.Name);
            Assert.DoesNotContain(log, l => l.Contains("must be named", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>A scheme with no windows is not a measurement - there is nothing in it to keep.</summary>
    [Fact]
    public void AWindowlessSchemeIsNotRecordedAsMeasured()
    {
        var catalog = new IsolationSchemeCatalog();
        catalog.AddMeasuredScheme(
            new IsolationScheme(IsolationScheme.ResultsOnlyName, Array.Empty<IsolationWindow>()),
            @"R:\cohort\run01.raw");

        Assert.Empty(catalog.Measured);
        Assert.True(catalog.IsEmpty);
    }

    /// <summary>
    /// A catalog holding only a measurement still saves. <c>IsEmpty</c> gated the write and counted
    /// documents and library entries only, so a directory whose sole knowledge came from the data
    /// files wrote nothing at all.
    /// </summary>
    [Fact]
    public void ADirectoryThatOnlyKnowsAMeasurementIsNotEmpty()
    {
        var catalog = new IsolationSchemeCatalog();
        catalog.AddMeasuredScheme(Measured(), @"R:\cohort\run01.raw");
        Assert.False(catalog.IsEmpty);
    }

    /// <summary>
    /// Files written before measurements were recorded still load. The scheme reached the library
    /// through <c>AddDocumentScheme</c> under a batch label that was really the scheme's name, so
    /// nothing about those files says "measured" - but the windows in them are still right.
    /// </summary>
    [Fact]
    public void AFileWrittenBeforeMeasurementsWereRecordedStillLoads()
    {
        var dir = TempDir();
        try
        {
            var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
            File.WriteAllText(path, """
                <prism_isolation_schemes>
                  <document batch="Imported from run01" scheme="Imported from run01">
                    <IsolationScheme name="Imported from run01">
                      <isolation_window start="400" end="403.0014" margin="0" />
                      <isolation_window start="403.0014" end="406.0028" margin="0" />
                    </IsolationScheme>
                  </document>
                </prism_isolation_schemes>
                """);

            var loaded = IsolationSchemeCatalog.Load(path);

            Assert.NotNull(loaded);
            Assert.Empty(loaded!.Measured);
            var offered = Assert.Single(loaded.UsableSchemes);
            Assert.Equal(2, offered.Windows.Count);
            Assert.False(loaded.IsMeasured(offered));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// Two cached schemes used to be an ambiguity the caller had to resolve by naming one. A cohort
    /// reaches that state honestly - Skyline declares a scheme and PRISM reads another out of the
    /// data, the same acquisition under two names - and the data's own answer is the one to take.
    /// </summary>
    [Fact]
    public void TheMeasuredSchemeWinsWhenTheCacheWouldOtherwiseBeAmbiguous()
    {
        var dir = TempDir();
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.AddDocumentScheme("Plate1", Declared());
            catalog.AddMeasuredScheme(Measured(), @"R:\cohort\run01.raw");
            catalog.Save(Path.Combine(dir, IsolationSchemeCatalog.FileName));

            var log = new System.Collections.Generic.List<string>();
            var resolved = IsolationSchemeResolver.Resolve(dir, rawDir: null, log.Add);

            Assert.NotNull(resolved);
            Assert.Equal("Imported from run01", resolved!.Name);
            Assert.Contains(log, l => l.Contains("measured from the data", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Two DECLARED schemes are still an ambiguity - neither is the acquisition's own answer.</summary>
    [Fact]
    public void TwoDeclaredSchemesAreStillAmbiguous()
    {
        var dir = TempDir();
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.AddDocumentScheme("Plate1", Declared());
            catalog.AddDocumentScheme("Plate2", new IsolationScheme("Other", new[]
            {
                new IsolationWindow(500, 520), new IsolationWindow(520, 540),
            }));
            catalog.Save(Path.Combine(dir, IsolationSchemeCatalog.FileName));

            var log = new System.Collections.Generic.List<string>();
            Assert.Null(IsolationSchemeResolver.Resolve(dir, rawDir: null, log.Add));
            Assert.Contains(log, l => l.Contains("must be named", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// The scheme reaches parameters.json - the file that travels with a result - complete enough to
    /// rebuild the grid from, and without disturbing what --from-provenance reads back.
    /// </summary>
    [Fact]
    public void TheSchemeIsRecordedInProvenanceWithoutDisturbingTheConfig()
    {
        var dir = TempDir();
        try
        {
            var config = new PrismConfig();
            config.BatchCorrection.Enabled = false;
            config.Parsimony.Enzyme = "lys-c";
            var jsonPath = Path.Combine(dir, Provenance.FileName);
            Provenance.Write(
                jsonPath, config, new[] { "plate1.csv" }, new Provenance.Stats(4, 100, 20, 20),
                "2026-09-12T04:00:00Z");

            var catalog = new IsolationSchemeCatalog();
            catalog.AddMeasuredScheme(Measured(), @"R:\cohort\run01.raw");

            Assert.True(Provenance.RecordIsolationSchemes(dir, catalog));

            using var doc = JsonDocument.Parse(File.ReadAllText(jsonPath));
            var entry = Assert.Single(doc.RootElement.GetProperty("isolation_schemes").EnumerateArray());
            Assert.Equal("measured", entry.GetProperty("source").GetString());
            Assert.Equal(@"R:\cohort\run01.raw", entry.GetProperty("data_file").GetString());
            Assert.Equal(3, entry.GetProperty("window_count").GetInt32());
            Assert.Equal(400.0, entry.GetProperty("mz_start").GetDouble(), 6);
            Assert.Equal(409.0042, entry.GetProperty("mz_end").GetDouble(), 6);
            Assert.False(entry.GetProperty("scheduled").GetBoolean());

            // The edges, not just a description: a summary tells a reader what the acquisition was,
            // the edges let a program bin on it. Both, because both questions get asked of this file.
            var windows = entry.GetProperty("windows").EnumerateArray().ToList();
            Assert.Equal(3, windows.Count);
            Assert.Equal(403.0014, windows[0].GetProperty("end").GetDouble(), 6);
            Assert.Contains("3 windows", entry.GetProperty("summary").GetString());

            // Untouched: this is the half of the file that must keep round-tripping.
            var reloaded = Provenance.LoadConfig(jsonPath);
            Assert.False(reloaded.BatchCorrection.Enabled);
            Assert.Equal("lys-c", reloaded.Parsimony.Enzyme);
            Assert.Equal(
                "2026-09-12T04:00:00Z", Provenance.ReadRunInfo(jsonPath)!.ProcessingDate);

            var summary = Assert.Single(Provenance.IsolationSchemeSummaries(jsonPath));
            Assert.Contains("measured from the data", summary);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A scheduled scheme keeps its firing intervals through provenance. Nothing produces one today,
    /// but recording a dynamic-DIA acquisition without the RT dimension would record a grid that was
    /// never acquired all at once.
    /// </summary>
    [Fact]
    public void ScheduledWindowsKeepTheirIntervals()
    {
        var dir = TempDir();
        try
        {
            var jsonPath = Path.Combine(dir, Provenance.FileName);
            Provenance.Write(
                jsonPath, new PrismConfig(), Array.Empty<string>(),
                new Provenance.Stats(0, 0, 0, 0), "2026-09-12T04:00:00Z");

            var catalog = new IsolationSchemeCatalog();
            catalog.AddMeasuredScheme(
                new IsolationScheme("dynamic", new[]
                {
                    new IsolationWindow(400, 410, 0.5, 10.0, 20.0),
                    new IsolationWindow(410, 420, 0.5, 20.0, 30.0),
                }),
                @"R:\cohort\run01.raw");

            Assert.True(Provenance.RecordIsolationSchemes(dir, catalog));

            using var doc = JsonDocument.Parse(File.ReadAllText(jsonPath));
            var entry = Assert.Single(doc.RootElement.GetProperty("isolation_schemes").EnumerateArray());
            Assert.True(entry.GetProperty("scheduled").GetBoolean());
            var first = entry.GetProperty("windows").EnumerateArray().First();
            Assert.Equal(0.5, first.GetProperty("margin").GetDouble(), 6);
            Assert.Equal(10.0, first.GetProperty("rt_start").GetDouble(), 6);
            Assert.Equal(20.0, first.GetProperty("rt_stop").GetDouble(), 6);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A document that declares real windows is worth recording too - but "Results only", which is
    /// what a DIA document normally says, has no geometry to preserve and contributes nothing.
    /// </summary>
    [Fact]
    public void DeclaredWindowsAreRecordedAndResultsOnlyIsNot()
    {
        var dir = TempDir();
        try
        {
            var jsonPath = Path.Combine(dir, Provenance.FileName);
            Provenance.Write(
                jsonPath, new PrismConfig(), Array.Empty<string>(),
                new Provenance.Stats(0, 0, 0, 0), "2026-09-12T04:00:00Z");

            var catalog = new IsolationSchemeCatalog();
            catalog.AddDocumentScheme("Plate1", Declared());
            catalog.SetAcquisition("Plate1", "DIA");
            catalog.AddDocumentScheme(
                "Plate2",
                new IsolationScheme(IsolationScheme.ResultsOnlyName, Array.Empty<IsolationWindow>()));

            Assert.True(Provenance.RecordIsolationSchemes(dir, catalog));

            using var doc = JsonDocument.Parse(File.ReadAllText(jsonPath));
            var entry = Assert.Single(doc.RootElement.GetProperty("isolation_schemes").EnumerateArray());
            Assert.Equal("document", entry.GetProperty("source").GetString());
            Assert.Equal("Plate1", entry.GetProperty("batch").GetString());
            Assert.Equal("DIA", entry.GetProperty("acquisition").GetString());
            Assert.Equal(2, entry.GetProperty("window_count").GetInt32());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// No provenance file, nothing to enrich - and no file created. A directory the density tab was
    /// merely pointed at is not a run, and inventing a parameters.json there would make it look like
    /// one to every reader that checks for the file.
    /// </summary>
    [Fact]
    public void NoProvenanceFileMeansNothingIsWritten()
    {
        var dir = TempDir();
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.AddMeasuredScheme(Measured(), @"R:\cohort\run01.raw");

            Assert.False(Provenance.RecordIsolationSchemes(dir, catalog));
            Assert.False(File.Exists(Path.Combine(dir, Provenance.FileName)));
            Assert.Empty(Provenance.IsolationSchemeSummaries(Path.Combine(dir, Provenance.FileName)));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
