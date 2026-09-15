using System;
using System.IO;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using FastaArchive = SkylinePrism.Core.Pipeline.FastaArchive;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// What a run is about to write over, and whether it would be writing anything different.
/// </summary>
/// <remarks>
/// Pointing PRISM at a directory that already held a cohort's results said nothing at all: it
/// recomputed what the settings changed and replaced the rest without a word. The hard part is not
/// noticing the files - it is staying quiet on the ordinary case, because re-running to regenerate a
/// report or top up a partial ion accounting is normal, and a warning that fires every time stops
/// being read.
/// </remarks>
public class ExistingResultsTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-existing-" + Guid.NewGuid().ToString("N"));

    public ExistingResultsTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Fact]
    public void AnEmptyDirectoryHasNothingToSay()
    {
        var existing = ExistingResults.Inspect(_dir, new PrismConfig());
        Assert.False(existing.Any);
        Assert.False(existing.WouldReplace);
        Assert.Null(existing.Warning());
    }

    [Fact]
    public void ADirectoryThatDoesNotExistHasNothingToSay()
    {
        var existing = ExistingResults.Inspect(Path.Combine(_dir, "nope"), new PrismConfig());
        Assert.False(existing.Any);
        Assert.Null(existing.Warning());
    }

    /// <summary>
    /// The case that has to stay silent IN THE LOG, or the warning becomes noise: same version, same
    /// settings, so nothing different would be written.
    /// </summary>
    /// <remarks>
    /// The person is asked anyway - see <see cref="AFinishedAnalysisIsWorthAskingAboutEveryTime"/>.
    /// The two questions are different: whether the numbers would differ, and whether files someone
    /// wants are about to be deleted.
    /// </remarks>
    [Fact]
    public void TheSameVersionAndSettingsIsSilent()
    {
        var config = new PrismConfig();
        Results(config);

        var existing = ExistingResults.Inspect(_dir, config);
        Assert.True(existing.Any);
        Assert.True(existing.SameVersion);
        Assert.True(existing.SameSettings);
        Assert.False(existing.WouldReplace);
        Assert.Null(existing.Warning());
    }

    /// <summary>
    /// A finished analysis in the folder is asked about every time, however ordinary the run.
    /// </summary>
    /// <remarks>
    /// The check used to fire only when this run would produce something DIFFERENT from what was
    /// there, which is the wrong question for a person: a re-run with identical settings still
    /// deletes and rewrites every file, and if those files are a colleague's finished analysis they
    /// are just as gone. The default output directory is the document folder's PRISM-Output, so
    /// landing on a previous analysis takes no mistake at all.
    /// </remarks>
    [Fact]
    public void AFinishedAnalysisIsWorthAskingAboutEveryTime()
    {
        var config = new PrismConfig();
        Results(config);

        var existing = ExistingResults.Inspect(_dir, config);

        // Nothing DIFFERENT would be written - and the person is still asked.
        Assert.Null(existing.Warning());
        var prompt = existing.OverwritePrompt()!;
        Assert.Contains("already holds a finished analysis", prompt, StringComparison.Ordinal);
        Assert.Contains("overwrites it", prompt, StringComparison.Ordinal);
        Assert.Contains("corrected_peptides.parquet", prompt, StringComparison.Ordinal);
    }

    /// <summary>An empty directory is not a question.</summary>
    [Fact]
    public void NothingToOverwriteAsksNothing()
    {
        Assert.Null(ExistingResults.Inspect(_dir, new PrismConfig()).OverwritePrompt());
        Assert.Null(ExistingResults.Inspect(
            Path.Combine(_dir, "nope"), new PrismConfig()).OverwritePrompt());
    }

    /// <summary>
    /// Results from another computer say so, because that is the difference between overwriting your
    /// own re-run and overwriting a colleague's cohort. Your own machine is not named - that is noise.
    /// </summary>
    [Fact]
    public void ResultsFromAnotherComputerNameTheMachine()
    {
        var config = new PrismConfig();
        Results(config);
        Assert.DoesNotContain(
            Environment.MachineName, ExistingResults.Inspect(_dir, config).OverwritePrompt()!,
            StringComparison.OrdinalIgnoreCase);

        SetRecordedHost("SCARFELL");

        var prompt = ExistingResults.Inspect(_dir, config).OverwritePrompt()!;
        Assert.Contains("SCARFELL", prompt, StringComparison.Ordinal);
    }

    /// <summary>
    /// A cohort written as tsv names its own outputs, not just the two files whose extension does not
    /// depend on <c>output.format</c>.
    /// </summary>
    /// <remarks>
    /// The directory was always recognized as holding results - protein_groups.csv is rewritten on
    /// every run whatever the format - but the question named it and the report while leaving out the
    /// two files someone actually minds losing.
    /// </remarks>
    [Fact]
    public void ATsvAnalysisNamesItsOwnOutputs()
    {
        var config = new PrismConfig();
        Results(config);
        File.Delete(Path.Combine(_dir, "corrected_peptides.parquet"));
        File.Delete(Path.Combine(_dir, "corrected_proteins.parquet"));
        File.WriteAllText(Path.Combine(_dir, "corrected_peptides.tsv"), "x");
        File.WriteAllText(Path.Combine(_dir, "corrected_proteins.tsv"), "x");

        var prompt = ExistingResults.Inspect(_dir, config).OverwritePrompt()!;
        Assert.Contains("corrected_peptides.tsv", prompt, StringComparison.Ordinal);
        Assert.Contains("corrected_proteins.tsv", prompt, StringComparison.Ordinal);
    }

    /// <summary>
    /// A provenance field of the wrong type is ignored, not thrown over.
    /// </summary>
    /// <remarks>
    /// <c>JsonElement.GetString()</c> throws on a value of another kind, and
    /// <c>InvalidOperationException</c> is not among the exceptions Inspect catches - so a numeric
    /// host aborted the whole pre-run check, and with it the run, over a field used for nothing but a
    /// sentence.
    /// </remarks>
    [Fact]
    public void AProvenanceFieldOfTheWrongTypeIsIgnored()
    {
        var config = new PrismConfig();
        Results(config);
        var path = Path.Combine(_dir, Provenance.FileName);
        var json = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path))!;
        json["host"] = 123;
        json["pipeline_version"] = true;
        File.WriteAllText(path, json.ToJsonString());

        var existing = ExistingResults.Inspect(_dir, config);

        var prompt = existing.OverwritePrompt()!;
        Assert.Contains("already holds a finished analysis", prompt, StringComparison.Ordinal);
        Assert.DoesNotContain("123", prompt, StringComparison.Ordinal);
    }

    /// <summary>Rewrite the recorded host, as a run on another computer would have left it.</summary>
    private void SetRecordedHost(string host)
    {
        var path = Path.Combine(_dir, Provenance.FileName);
        var json = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path))!;
        json["host"] = host;
        File.WriteAllText(path, json.ToJsonString());
    }

    [Fact]
    public void DifferentSettingsAreReported()
    {
        Results(new PrismConfig());

        var changed = new PrismConfig();
        changed.TransitionRollup.Method = "median_polish";

        var existing = ExistingResults.Inspect(_dir, changed);
        Assert.True(existing.SameVersion);
        Assert.False(existing.SameSettings);
        Assert.True(existing.WouldReplace);

        var warning = existing.Warning()!;
        Assert.Contains("different settings", warning, StringComparison.Ordinal);
        Assert.Contains("corrected_peptides.parquet", warning, StringComparison.Ordinal);
        // It has to say that unchanged stages are reused, or it reads as "everything is lost".
        Assert.Contains("reused", warning, StringComparison.Ordinal);
    }

    /// <summary>
    /// A setting no stage reads changes nothing, so it says nothing.
    /// </summary>
    /// <remarks>
    /// The first version of this compared the whole config's YAML, so ANY key that moved produced a
    /// warning that a cohort's results would be replaced - including keys StageDependencies lists as
    /// output-irrelevant. <c>processing.n_workers</c> is thread count; the rollup is asserted
    /// reproducible run to run, so it cannot move a number. Warning about it is how a warning stops
    /// being read.
    /// </remarks>
    [Fact]
    public void ASettingNoStageReadsIsSilent()
    {
        var config = new PrismConfig();
        Results(config);

        var faster = new PrismConfig();
        faster.Processing.NWorkers = 8;
        faster.QcReport.SavePlots = !faster.QcReport.SavePlots;

        var existing = ExistingResults.Inspect(_dir, faster);
        Assert.True(existing.SameSettings);
        Assert.Empty(existing.Recomputed);
        Assert.Null(existing.Warning());
    }

    /// <summary>
    /// A changed setting names the stages it changes, not every stage.
    /// </summary>
    [Fact]
    public void OnlyTheStagesBelowTheChangeAreListed()
    {
        Results(new PrismConfig());

        var changed = new PrismConfig();
        changed.ProteinRollup.Method = "sum";

        var existing = ExistingResults.Inspect(_dir, changed);

        // The protein rollup and what follows it. The transition rollup and peptide normalization
        // are upstream of the change and keep their outputs.
        Assert.Contains("the protein rollup", existing.Warning()!, StringComparison.Ordinal);
        Assert.DoesNotContain("the transition rollup", existing.Warning()!, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "peptide normalization", existing.Warning()!, StringComparison.Ordinal);
    }

    /// <summary>
    /// Results with no provenance beside them cannot be compared, so they are reported rather than
    /// assumed to match - the safe direction when the question is whether a cohort survives.
    /// </summary>
    [Fact]
    public void ResultsWithNoProvenanceAreReported()
    {
        File.WriteAllText(Path.Combine(_dir, "corrected_peptides.parquet"), "x");

        var existing = ExistingResults.Inspect(_dir, new PrismConfig());
        Assert.True(existing.WouldReplace);
        Assert.Contains("a previous run", existing.Warning()!, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnreadableProvenanceIsReportedRatherThanTrusted()
    {
        Results(new PrismConfig());
        File.WriteAllText(Path.Combine(_dir, "parameters.json"), "{ not json");

        Assert.True(ExistingResults.Inspect(_dir, new PrismConfig()).WouldReplace);
    }

    /// <summary>
    /// Working files are not results. A directory holding only a merge and a stage cache is one a
    /// previous run left partway, and warning about it would train people to ignore the dialog.
    /// </summary>
    [Fact]
    public void IntermediatesAloneAreNotResults()
    {
        File.WriteAllText(Path.Combine(_dir, "peptides_rollup.parquet"), "x");
        File.WriteAllText(Path.Combine(_dir, "stage_cache.json"), "{}");
        Directory.CreateDirectory(Path.Combine(_dir, "merged_data"));

        Assert.False(ExistingResults.Inspect(_dir, new PrismConfig()).Any);
    }

    /// <summary>
    /// The case that made the comparison worth getting right: a run whose FASTA has since moved.
    /// </summary>
    /// <remarks>
    /// <c>Provenance.LoadConfig</c> exists for RE-RUNNING a recorded config, so when the original
    /// database is gone it redirects <c>parsimony.fasta_path</c> to the copy the run archived beside
    /// its outputs. Comparing through it would report "different settings" for the very config that
    /// produced these results - and only once the archive had become load-bearing, which is the one
    /// time nobody wants a spurious prompt. The comparison reads what was RECORDED instead.
    /// </remarks>
    [Fact]
    public void AnArchivedFastaDoesNotMakeAnIdenticalRunLookDifferent()
    {
        var config = new PrismConfig
        {
            Parsimony = { FastaPath = Path.Combine(_dir, "gone", "human.fasta") },
        };

        var archive = Path.Combine(_dir, "fasta");
        Directory.CreateDirectory(archive);
        File.WriteAllText(Path.Combine(archive, "human.fasta"), ">sp|P1|A\nPEPTIDER\n");

        File.WriteAllText(Path.Combine(_dir, "corrected_peptides.parquet"), "x");
        Provenance.Write(
            Path.Combine(_dir, Provenance.FileName), config, new[] { "report.csv" },
            new Provenance.Stats(1, 10, 5, 5), "2026-01-01T00:00:00.0000000Z",
            new[]
            {
                new FastaArchive.Entry(
                    "parsimony.fasta_path", config.Parsimony.FastaPath!,
                    Path.Combine("fasta", "human.fasta")),
            });

        // The original is not on disk, so LoadConfig - the re-run path - redirects to the copy.
        // That is the behavior this must not be built on.
        Assert.Equal(
            Path.GetFullPath(Path.Combine(archive, "human.fasta")),
            Provenance.LoadConfig(Path.Combine(_dir, Provenance.FileName)).Parsimony.FastaPath);

        var existing = ExistingResults.Inspect(_dir, config);
        Assert.True(existing.SameSettings);
        Assert.Null(existing.Warning());
    }

    /// <summary>
    /// An input path the filesystem rejects does not take the run down with it.
    /// </summary>
    /// <remarks>
    /// Stamping the input files touches the filesystem, and this check runs BEFORE the pipeline -
    /// whose job it is to report a bad input properly. A courtesy warning must never be the thing
    /// that fails the run, so a comparison that cannot be made is reported rather than thrown.
    /// </remarks>
    [Fact]
    public void AnInputPathThatCannotBeStampedIsReportedNotThrown()
    {
        var config = new PrismConfig();
        Results(config);

        var existing = ExistingResults.Inspect(
            _dir, config, new[] { "no:such|file .csv", Path.Combine(_dir, "missing.csv") });

        // Cannot tell, so it says something rather than reassuring.
        Assert.True(existing.WouldReplace);
        Assert.NotNull(existing.Warning());
    }

    /// <summary>
    /// Without a stage cache the answer is a prediction, and the warning says so.
    /// </summary>
    [Fact]
    public void AWarningWithoutAStageCacheSaysItIsALowerBound()
    {
        Results(new PrismConfig());
        var changed = new PrismConfig();
        changed.ProteinRollup.Method = "sum";

        // No stage_cache.json here, so the transition rollup was never actually checked.
        Assert.Contains(
            "lower bound", ExistingResults.Inspect(_dir, changed).Warning()!, StringComparison.Ordinal);
    }

    /// <summary>Writes what a completed run leaves: the reported outputs plus its provenance.</summary>
    private void Results(PrismConfig config)
    {
        File.WriteAllText(Path.Combine(_dir, "corrected_peptides.parquet"), "x");
        File.WriteAllText(Path.Combine(_dir, "corrected_proteins.parquet"), "x");
        File.WriteAllText(Path.Combine(_dir, "qc_report.html"), "<html></html>");
        Provenance.Write(
            Path.Combine(_dir, Provenance.FileName), config, new[] { "report.csv" },
            new Provenance.Stats(1, 10, 5, 5), "2026-01-01T00:00:00.0000000Z");
    }
}
