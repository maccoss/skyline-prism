using System;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Tests for <see cref="QuantCommand"/>: what it refuses to express, how it quotes, and that it names
/// only what the analysis actually ran. Running the command it builds through the CLI - and comparing
/// every file - is <c>CliIntegrationTests.Differential_Report_FromTheButtonsRequest_IsReproducedByItsRecordedCommand</c>.
/// </summary>
public class QuantCommandTests
{
    private static QuantRequest Request(string outputDir, string[] a, string[] b,
        DifferentialOptions? options = null, ProteinList[]? panels = null, string? markerGroupBy = null,
        string? subjectColumn = null)
    {
        var ds = DifferentialDataset.Load(Fixtures.Path2("mini", "e2e-sum", "output"), FeatureLevel.Protein);
        return new QuantRequest
        {
            OutputDir = outputDir,
            Dataset = ds,
            Options = options ?? new DifferentialOptions(),
            Rule = SignificanceRule.Default,
            GroupBy = "sample_type",
            ALevels = a,
            BLevels = b,
            MarkerPanels = panels ?? Array.Empty<ProteinList>(),
            MarkerGroupBy = markerGroupBy,
            SubjectColumn = subjectColumn,
        };
    }

    [Theory]
    [InlineData("AD, early", "contains a comma")]
    [InlineData(" AD", "leading or trailing spaces")]
    [InlineData("-1", "starts with '-'")]
    [InlineData("AD $HOME", "no shell quotes the same way")]
    // Backslash, only where the two shells disagree: inside bash double quotes a doubled backslash
    // collapses to one while PowerShell keeps both, and a trailing one escapes the closing quote.
    [InlineData(@"AD\\early", "doubled backslash")]
    [InlineData(@"AD\", "ends with a backslash")]
    public void ALevelTheCommandLineCannotCarry_GivesNoCommand(string level, string why)
    {
        var request = Request("C:\\runs\\out", new[] { "Control" }, new[] { level });

        Assert.False(QuantCommand.TryArguments(request, out _, out var reason));
        Assert.Contains(why, reason);
        Assert.Null(QuantCommand.For(request, request.Options, out _));
    }

    [Fact]
    public void AUncPathGivesNoCommand_ButAnOrdinaryWindowsPathStillDoes()
    {
        // The case this came from: a run on a lab share. Quoted, bash collapses the leading "\\" to one
        // and hands the program a path that does not exist, while PowerShell passes it through - so the
        // printed line would mean two different things. Refused rather than offered wrong.
        var unc = Request(@"\\lab-nas\studies\out", new[] { "qc" }, new[] { "experimental" });
        Assert.False(QuantCommand.TryArguments(unc, out _, out var reason));
        Assert.Contains("doubled backslash", reason);

        // A mapped drive has only single backslashes before ordinary characters, which are literal in
        // both shells. Refusing every backslash would mean never offering a command on Windows.
        var mapped = Request(@"R:\studies\out", new[] { "qc" }, new[] { "experimental" });
        Assert.True(QuantCommand.TryArguments(mapped, out _, out _));
        Assert.Contains(@"""R:\studies\out""", QuantCommand.For(mapped, mapped.Options, out _));
    }

    [Fact]
    public void TheLineIsRenderedFromTheArgumentsItRecords()
    {
        // The report stores both; built from one resolution they cannot be two different answers.
        var request = Request(@"C:\runs\out", new[] { "qc" }, new[] { "experimental" });
        Assert.True(QuantCommand.TryArguments(request, out var args, out _));
        Assert.Equal(QuantCommand.For(request, request.Options, out _), QuantCommand.Line(args));
    }

    [Fact]
    public void APathIsAlwaysQuoted_AndLosesATrailingSeparator()
    {
        // Unquoted, bash reads each backslash as an escape; a trailing one would escape the closing quote.
        // Built with this platform's separator so the test means the same on Windows and Linux CI.
        var dir = System.IO.Path.Combine("runs", "out");
        var command = QuantCommand.For(
            Request(dir + System.IO.Path.DirectorySeparatorChar, new[] { "qc" }, new[] { "experimental" }),
            new DifferentialOptions(), out _)!;

        Assert.Contains($"-d \"{dir}\"", command);
        // A plain word needs no quotes; a comma list does, or PowerShell reads it as an array.
        Assert.Contains("--group-by sample_type", command);
        Assert.Contains("-a qc ", command);
    }

    [Fact]
    public void ACommaListIsQuoted()
    {
        var command = QuantCommand.For(Request("out", new[] { "qc", "reference" }, new[] { "experimental" }),
            new DifferentialOptions(), out _)!;
        Assert.Contains("-a \"qc,reference\"", command);
    }

    /// <summary>The command names the options the analysis RAN with, not the ones the pane still shows.</summary>
    [Fact]
    public void StaleCovariatesUnderWelch_AreNotInTheCommand()
    {
        var ds = DifferentialDataset.Load(Fixtures.Path2("mini", "e2e-sum", "output"), FeatureLevel.Protein);
        var stale = new DifferentialOptions
        {
            Test = DifferentialTest.WelchT,
            Covariates = new[] { Covariate.FromMetadata("batch", ds.MetadataValues("batch")) },
        };

        var effective = QuantAnalysis.EffectiveOptions(stale, out var note);

        Assert.Null(effective.Covariates);
        Assert.StartsWith("Not adjusted for batch", note);
        Assert.True(QuantCommand.TryArguments(Request("out", new[] { "qc" }, new[] { "experimental" }), effective,
            out var args, out _));
        Assert.DoesNotContain("--adjust-for", args);
        Assert.Contains("welch", args);
    }

    /// <summary>The blocked design records its subject column, or the command would run unpaired.</summary>
    [Fact]
    public void ABlockedDesign_RecordsItsSubjectColumn()
    {
        var blocked = new DifferentialOptions { Design = DifferentialDesign.BlockedBySubject };
        var request = Request("out", new[] { "qc" }, new[] { "experimental" }, blocked, subjectColumn: "batch");

        Assert.True(QuantCommand.TryArguments(request, out var args, out _));
        var list = args.ToList();
        Assert.Equal("blocked", list[list.IndexOf("--design") + 1]);
        Assert.Equal("batch", list[list.IndexOf("--subject") + 1]);
    }

    /// <summary>
    /// --markers is emitted only with a column to group by: a trend with none wrote no marker section,
    /// and the CLI refuses --markers on a trend without --markers-group-by.
    /// </summary>
    [Fact]
    public void MarkersOnATrendWithNoGroupColumn_AreLeftOut()
    {
        var panel = ProteinListSet.Load().WithBuiltIns().First(l => l.Name == ProteinList.EvMarkersName);
        var trend = new DifferentialOptions { Design = DifferentialDesign.LinearTrend, TrendColumn = "batch" };

        Assert.True(QuantCommand.TryArguments(
            Request("out", Array.Empty<string>(), Array.Empty<string>(), trend, new[] { panel }), out var none, out _));
        Assert.DoesNotContain("--markers", none);

        Assert.True(QuantCommand.TryArguments(
            Request("out", Array.Empty<string>(), Array.Empty<string>(), trend, new[] { panel }, "sample_type"),
            out var grouped, out _));
        Assert.Contains("--markers", grouped);
        Assert.Contains("--markers-group-by", grouped);
    }

    /// <summary>The test suite runs against the shipped panels only (see <see cref="HermeticEnvironment"/>).</summary>
    [Fact]
    public void TheListsPathOverride_IsHonoured()
    {
        Assert.Equal(Environment.GetEnvironmentVariable(ProteinListSet.ListsPathVariable), ProteinListSet.DefaultPath);
        Assert.False(System.IO.File.Exists(ProteinListSet.DefaultPath));
    }
}
