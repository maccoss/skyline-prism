using System;
using System.IO;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The QC report's "Analysis Information" block describes the RUN, not the invocation that rendered
/// it. These cover the three ways that can go wrong: no provenance at all, an unreadable one, and a
/// `prism qc -c` whose config is not the config the run used.
/// </summary>
public class QcReportProvenanceTests : IDisposable
{
    private readonly string _dir;

    public QcReportProvenanceTests()
    {
        // One real run, reused by every case (the pipeline is the expensive part).
        _dir = Path.Combine(Path.GetTempPath(), "prism_qcprov_" + Guid.NewGuid().ToString("N"));
        var mergeDir = Fixtures.Path2("mini", "merge");
        var inputs = new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };
        var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-sum"), "config.yaml"));
        config.QcReport.Enabled = false; // generated explicitly per case
        PrismPipeline.Run(inputs, _dir, config);
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    private string ProvenancePath => Path.Combine(_dir, "parameters.json");

    private string Generate(PrismConfig config) =>
        File.ReadAllText(QcReport.Generate(_dir, config, savePlots: false));

    [Fact]
    public void ReportsTheRunsSettings_NotTheOnesPassedIn()
    {
        // The run used method: sum. A caller passing something else - `prism qc -c qc_only.yaml`, whose
        // omitted sections default to median_polish - must not have that printed as what happened.
        var runMethod = Provenance.LoadConfig(ProvenancePath).TransitionRollup.Method;
        Assert.Equal("sum", runMethod);

        var other = new PrismConfig();
        other.TransitionRollup.Method = "median_polish";
        var html = Generate(other);

        Assert.Contains("method=sum", html);
        Assert.DoesNotContain("method=median_polish, min_transitions", html);
    }

    [Fact]
    public void WithoutProvenance_StillRendersAndSaysTheVersionIsUnrecorded()
    {
        var saved = File.ReadAllText(ProvenancePath);
        File.Delete(ProvenancePath);
        try
        {
            var html = Generate(new PrismConfig());

            Assert.Contains("PRISM QC Report", html);
            Assert.Contains("unrecorded", html);
            // No run facts are invented for a directory that records none.
            Assert.DoesNotContain("Processing date", html);
            Assert.DoesNotContain("Computer", html);
        }
        finally
        {
            File.WriteAllText(ProvenancePath, saved);
        }
    }

    [Fact]
    public void WithAMalformedProvenance_StillRendersRatherThanThrowing()
    {
        var saved = File.ReadAllText(ProvenancePath);
        File.WriteAllText(ProvenancePath, "{ this is not json");
        try
        {
            var html = Generate(new PrismConfig());
            Assert.Contains("PRISM QC Report", html);
            Assert.Contains("unrecorded", html);
        }
        finally
        {
            File.WriteAllText(ProvenancePath, saved);
        }
    }

    [Fact]
    public void RecordsTheRunningMachine_SoARegeneratedReportDoesNotClaimTheRendererHost()
    {
        var info = Provenance.ReadRunInfo(ProvenancePath);
        Assert.NotNull(info);
        Assert.Equal(Environment.MachineName, info!.Host);
        Assert.Contains(Environment.MachineName, Generate(new PrismConfig()));
    }

    /// <summary>
    /// The acquisition's window layout reaches the page that gets kept. Once the instrument files are
    /// deleted - the normal end of an analysis - the report and the provenance file beside it are the
    /// only things left that can say what the data was acquired with.
    /// </summary>
    [Fact]
    public void NamesTheAcquisitionWhenTheRunRecordedOne()
    {
        var saved = File.ReadAllText(ProvenancePath);
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.AddMeasuredScheme(
                new IsolationScheme("Imported from run01", new[]
                {
                    new IsolationWindow(400.0, 403.0014),
                    new IsolationWindow(403.0014, 406.0028),
                }),
                @"R:\cohort\run01.raw");
            Assert.True(Provenance.RecordIsolationSchemes(_dir, catalog));

            var html = Generate(new PrismConfig());

            Assert.Contains("Isolation scheme", html);
            Assert.Contains("2 windows", html);
            // Where the windows came from is the basis for trusting them, so it is on the page too.
            Assert.Contains("measured from the data", html);
        }
        finally
        {
            File.WriteAllText(ProvenancePath, saved);
        }
    }

    /// <summary>
    /// A run whose data files were never reachable records no scheme, and the report must not grow an
    /// empty row implying one was found.
    /// </summary>
    [Fact]
    public void SaysNothingAboutTheAcquisitionWhenNoneWasRecorded()
    {
        Assert.Empty(Provenance.IsolationSchemeSummaries(ProvenancePath));
        Assert.DoesNotContain("Isolation scheme", Generate(new PrismConfig()));
    }
}
