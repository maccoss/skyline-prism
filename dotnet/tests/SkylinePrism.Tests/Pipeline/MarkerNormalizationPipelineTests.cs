using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// Marker normalization end to end: it runs after both arms, adjusts BOTH corrected outputs with one
/// protein-level score, keeps the markers and flags them, and leaves the pre-correction intermediates
/// alone.
/// </summary>
public class MarkerNormalizationPipelineTests : IDisposable
{
    private readonly string _dir;
    private readonly string _listFile;

    public MarkerNormalizationPipelineTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "prism_marker_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        // The mini fixture's three protein groups; two of them are the "markers" here.
        _listFile = Path.Combine(_dir, "markers.txt");
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    private static string[] Inputs()
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        return new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };
    }

    private static PrismConfig Config()
    {
        var c = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-medpolish"), "config.yaml"));
        c.QcReport.Enabled = false;
        c.ProteinRollup.MinPeptides = 1;
        return c;
    }

    private string RunOnce(Action<PrismConfig>? tweak = null)
    {
        var outDir = Path.Combine(_dir, "run_" + Guid.NewGuid().ToString("N")[..6]);
        var config = Config();
        tweak?.Invoke(config);
        PrismPipeline.Run(Inputs(), outDir, config, null, _ => { });
        return outDir;
    }

    /// <summary>
    /// The mini fixture has exactly three protein groups and a score needs at least three markers, so
    /// here the marker set is the whole set. That still exercises everything structural: both outputs
    /// adjusted from one protein-level score, the markers flagged, the score recorded, and the protein
    /// arm's INPUT left alone.
    /// </summary>
    [Fact]
    public void ItAdjustsBothCorrectedOutputs_FlagsTheMarkers_AndLeavesIntermediatesAlone()
    {
        var plain = RunOnce();
        File.WriteAllLines(_listFile, AllProteins(Path.Combine(plain, "corrected_proteins.parquet")));

        var adjusted = RunOnce(c =>
        {
            c.MarkerNormalization.Enabled = true;
            c.MarkerNormalization.ProteinListFile = _listFile;
        });

        using (var prot = ParquetColumnReader.Open(Path.Combine(adjusted, "corrected_proteins.parquet")))
        {
            Assert.True(prot.HasColumn(MarkerNormalizeStage.MarkerColumn));
            Assert.Equal(3, prot.ReadDoubles(MarkerNormalizeStage.MarkerColumn).Count(f => f != 0));
            // The metadata the file carried is still there - this stage rewrites the file, and a
            // projection of it would quietly drop the protein-group columns everything else joins on.
            Assert.True(prot.HasColumn("leading_gene_name"));
            Assert.True(prot.HasColumn("n_peptides"));
        }
        using (var pep = ParquetColumnReader.Open(Path.Combine(adjusted, "corrected_peptides.parquet")))
        {
            Assert.True(pep.HasColumn(MarkerNormalizeStage.MarkerColumn));
            Assert.Contains(pep.ReadDoubles(MarkerNormalizeStage.MarkerColumn), f => f != 0);
            Assert.True(pep.HasColumn("protein_group"));
        }

        var scoreCsv = Path.Combine(adjusted, "marker_normalization.csv");
        Assert.True(File.Exists(scoreCsv));
        Assert.Equal(
            new[] { "sample_id", "marker_score" },
            CsvLine.Split(File.ReadAllLines(scoreCsv)[0]));

        // peptides_log2_internal is what the protein rollup consumed, produced before the score
        // existed. Adjusting it would describe a rollup that never happened.
        Assert.Equal(
            Hash(Path.Combine(plain, "peptides_log2_internal.parquet")),
            Hash(Path.Combine(adjusted, "peptides_log2_internal.parquet")));
    }

    [Fact]
    public void TheMarkerAxisIsRemoved_SoEachMarkerVariesLessThanBefore()
    {
        // The by-construction property: after residualising on a score built from these features,
        // what remains of each of them no longer carries the axis they shared.
        var plain = RunOnce();
        File.WriteAllLines(_listFile, AllProteins(Path.Combine(plain, "corrected_proteins.parquet")));

        var adjusted = RunOnce(c =>
        {
            c.MarkerNormalization.Enabled = true;
            c.MarkerNormalization.ProteinListFile = _listFile;
        });

        var before = LogVariancePerProtein(Path.Combine(plain, "corrected_proteins.parquet"));
        var after = LogVariancePerProtein(Path.Combine(adjusted, "corrected_proteins.parquet"));

        Assert.Equal(before.Length, after.Length);
        for (var i = 0; i < before.Length; i++)
            Assert.True(after[i] <= before[i] + 1e-9,
                $"protein {i}: variance rose from {before[i]:0.####} to {after[i]:0.####}");
        Assert.True(after.Sum() < before.Sum(),
            "residualising on the markers' own axis should reduce their spread");
    }

    private static string[] AllProteins(string path)
    {
        using var r = ParquetColumnReader.Open(path);
        return r.ReadStrings("leading_protein");
    }

    /// <summary>Across-sample variance of each protein's log2 profile.</summary>
    private static double[] LogVariancePerProtein(string path)
    {
        using var r = ParquetColumnReader.Open(path);
        var meta = new[] { "protein_group", "leading_protein", "leading_name", "leading_uniprot_id",
            "leading_gene_name", "leading_description", "n_peptides", "n_unique_peptides",
            "low_confidence", MarkerNormalizeStage.MarkerColumn };
        var cols = r.ColumnNames.Where(c => !meta.Contains(c)).Select(r.ReadDoubles).ToList();
        var result = new double[r.RowCount];
        for (var i = 0; i < r.RowCount; i++)
        {
            var values = cols.Select(c => Math.Log2(Math.Max(c[i], 1e-6)))
                .Where(v => !double.IsNaN(v)).ToList();
            var mean = values.Average();
            result[i] = values.Sum(v => (v - mean) * (v - mean)) / Math.Max(1, values.Count - 1);
        }
        return result;
    }

    [Fact]
    public void WithoutEnoughQuantifiedMarkers_ItRefusesRatherThanGuessing()
    {
        File.WriteAllLines(_listFile, new[] { "NOT_A_REAL_PROTEIN_1", "NOT_A_REAL_PROTEIN_2" });

        var ex = Assert.Throws<InvalidOperationException>(() => RunOnce(c =>
        {
            c.MarkerNormalization.Enabled = true;
            c.MarkerNormalization.ProteinListFile = _listFile;
        }));
        Assert.Contains("at least", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TheShippedEvPanelIsResolvableByName_WithoutAnySavedLists()
    {
        // A CLI run on a machine with no protein-lists.json must still find the panel PRISM ships.
        var list = ProteinListSet.Resolve(ProteinList.EvMarkersName, null, Path.Combine(_dir, "none.json"));
        Assert.NotNull(list);
        Assert.Equal(18, list!.Members.Count);
        Assert.Contains("CD9", list.Members);
        Assert.Contains("PDCD6IP", list.Members);
    }

    [Fact]
    public void AMissingNamedListIsAnError_NamingWhatIsAvailable()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            ProteinListSet.Resolve("no such list", null, Path.Combine(_dir, "none.json")));
        Assert.Contains(ProteinList.EvMarkersName, ex.Message);
    }

    private static byte[] Hash(string path) =>
        System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(path));
}
