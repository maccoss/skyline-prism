using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Where the theoretical ions become claims: <see cref="ClaimedRegionLoader"/> reading a merged
/// table that carries a precursor charge.
///
/// <para>The mass arithmetic has its own tests. What is pinned here is the WIRING - that a precursor
/// which reconciles contributes its ions, that one which does not contributes nothing and is
/// counted, and that an export without a charge column produces no explained set at all rather than
/// an empty one. The last two are the same safety property seen from either side, and they are what
/// stands between a heavy-labeled document and a peptide's worth of claims on m/z belonging to
/// nothing.</para>
/// </summary>
public class TheoreticalClaimTests
{
    private const string Peptide = "PEPTIDEK";
    private const int Charge = 2;

    /// <summary>The m/z Skyline would export for this precursor, from PRISM's own tables.</summary>
    private static double CorrectPrecursorMz =>
        PeptideFragments.PrecursorMz(Peptide, Charge)!.Value;

    [Fact]
    public void AReconcilingPrecursorContributesEveryTheoreticalIon()
    {
        var dir = TempDir();
        try
        {
            var loaded = Load(dir, CorrectPrecursorMz);

            Assert.Equal(1, loaded.Precursors);
            Assert.Equal(0, loaded.Unreconciled);
            Assert.True(loaded.HasExplained);

            // The quantified fragment is in the explained set too - the union half - so the count is
            // the theoretical ions plus that one claim.
            var theoretical = PeptideFragments.Enumerate(Peptide, Charge).Count;
            Assert.Equal(theoretical + 1, loaded.ExplainedRegions.Count);

            // ...and every explained region is MS2. MS1 is deliberately not enumerated.
            Assert.All(loaded.ExplainedRegions, r => Assert.Equal(2, r.MsLevel));
        }
        finally { Cleanup(dir); }
    }

    /// <summary>
    /// The safety property. A sequence whose mass PRISM cannot reproduce - the shape a heavy isotope
    /// label takes, since the exported sequence carries only structural modifications - claims
    /// NOTHING, and is counted so the gap is visible rather than showing up as a quietly smaller
    /// total.
    /// </summary>
    [Fact]
    public void APrecursorThatDoesNotReconcileClaimsNothingAndIsCounted()
    {
        var dir = TempDir();
        try
        {
            // +8 Da on the precursor, which is what a 13C6-15N2 lysine would do and what the
            // peptide-level sequence would not show.
            var loaded = Load(dir, CorrectPrecursorMz + (8.0143 / Charge));

            Assert.Equal(1, loaded.Precursors);
            Assert.Equal(1, loaded.Unreconciled);

            // Only the quantified fragment survives into the explained set; no theoretical ion does.
            Assert.Single(loaded.ExplainedRegions);
            Assert.Contains("NOT RECONCILED", loaded.Describe(), StringComparison.Ordinal);
        }
        finally { Cleanup(dir); }
    }

    /// <summary>
    /// No charge column, no explained set - and no PRETENCE of one. A zero here would be read as
    /// "these peptides account for nothing".
    /// </summary>
    [Fact]
    public void WithoutAChargeColumnThereIsNoExplainedSetAtAll()
    {
        var dir = TempDir();
        try
        {
            var loaded = Load(dir, CorrectPrecursorMz, withCharge: false);

            Assert.False(loaded.HasExplained);
            Assert.Empty(loaded.ExplainedRegions);
            Assert.Equal(0, loaded.Precursors);
            Assert.Equal(0, loaded.Unreconciled);
            Assert.DoesNotContain("explained", loaded.Describe(), StringComparison.OrdinalIgnoreCase);

            // The quantified accounting is unaffected - that is the whole point of it being optional.
            Assert.NotEmpty(loaded.Regions);
        }
        finally { Cleanup(dir); }
    }

    /// <summary>
    /// The predicate bounds the WORK. A sample the caller does not want must not have a claim set
    /// built for it at all - on a real cohort that is a multi-million-region array per skipped
    /// replicate, which is what made `--max 3` on 93 replicates so expensive.
    /// </summary>
    [Fact]
    public void AnUnwantedSampleIsNeverAccumulated()
    {
        var dir = TempDir();
        try
        {
            Write(dir, new[]
            {
                ("r1", CorrectPrecursorMz),
                ("r2", CorrectPrecursorMz),
                ("r3", CorrectPrecursorMz),
            });

            var (dataset, cols, scheme, tolerance, classes) = Setup(dir);
            var seen = new List<string>();

            ClaimedRegionLoader.ForEachSample(
                dataset, cols, scheme, tolerance, tolerance, classes,
                (sample, loaded) =>
                {
                    seen.Add(sample);
                    Assert.NotEmpty(loaded.ExplainedRegions);
                },
                memoryBudgetMb: 0,
                wanted: s => s == "r2");

            Assert.Equal(new[] { "r2" }, seen);
        }
        finally { Cleanup(dir); }
    }

    /// <summary>With no predicate every sample is delivered, which is the pre-existing behavior.</summary>
    [Fact]
    public void WithNoPredicateEverySampleIsDelivered()
    {
        var dir = TempDir();
        try
        {
            Write(dir, new[] { ("r1", CorrectPrecursorMz), ("r2", CorrectPrecursorMz) });
            var (dataset, cols, scheme, tolerance, classes) = Setup(dir);

            var seen = new List<string>();
            ClaimedRegionLoader.ForEachSample(
                dataset, cols, scheme, tolerance, tolerance, classes, (s, _) => seen.Add(s));

            Assert.Equal(new[] { "r1", "r2" }, seen);
        }
        finally { Cleanup(dir); }
    }

    // ------------------------------------------------------------------ harness

    private static ClaimedRegionLoader.Loaded Load(
        string dir, double precursorMz, bool withCharge = true)
    {
        Write(dir, new[] { ("r1", precursorMz) }, withCharge);
        var (dataset, cols, scheme, tolerance, classes) = Setup(dir);
        return ClaimedRegionLoader.ForReplicate(
            dataset, cols, "r1", scheme, tolerance, tolerance, classes);
    }

    private static (MergedDataset Dataset, SignalColumns Cols, IsolationScheme Scheme,
        ProductMassTolerance Tolerance, IReadOnlyDictionary<string, PeptideClass> Classes)
        Setup(string dir)
    {
        var parquet = Path.Combine(dir, "merged_data.parquet");
        var dataset = MergedDataset.Open(parquet);
        var cols = SignalColumns.Resolve(ParquetTable.ReadColumnNames(parquet).ToList())
            ?? throw new InvalidOperationException("columns did not resolve");
        var scheme = IsolationScheme.Cycle("c", start: 400, step: 200, width: 200, count: 2);
        var tolerance = ProductMassTolerance.ParseSetting("10 ppm")!;
        var classes = new Dictionary<string, PeptideClass> { [Peptide] = new(true, 0) };
        return (dataset, cols, scheme, tolerance, classes);
    }

    /// <summary>
    /// One fragment row per sample, carrying the precursor charge unless asked not to. The fragment
    /// m/z sits inside the isolation window so the claim resolves.
    /// </summary>
    private static void Write(
        string dir, IReadOnlyList<(string Sample, double PrecursorMz)> rows, bool withCharge = true)
    {
        // The single-file layout, which MergedDataset supports alongside the partitioned directory
        // and which keeps this fixture to one file.
        var path = Path.Combine(dir, "merged_data.parquet").Replace('\\', '/');

        var select = string.Join(" UNION ALL ", rows.Select(r => "SELECT "
            + $"'{r.Sample}' AS \"Sample ID\", "
            + $"'{Peptide}' AS \"Peptide Modified Sequence\", "
            + "'y5' AS \"Fragment Ion\", "
            + $"{Num(r.PrecursorMz)} AS \"Precursor Mz\", "
            + "600.0 AS \"Product Mz\", "
            + "10.0 AS \"Start Time\", "
            + "11.0 AS \"End Time\""
            + (withCharge ? $", {Charge} AS \"Precursor Charge\"" : "")));

        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $"COPY ({select}) TO '{path}' (FORMAT PARQUET)";
        cmd.ExecuteNonQuery();
    }

    private static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_theo_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        try { Directory.Delete(dir, recursive: true); }
        catch (IOException) { /* a temp directory that will not delete is not a failure */ }
    }
}
