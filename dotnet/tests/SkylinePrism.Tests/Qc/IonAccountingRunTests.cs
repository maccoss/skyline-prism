using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using DuckDB.NET.Data;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The orchestration: pairing replicates to files, keying the cache, reusing what it covers,
/// measuring the rest, and writing progress as it goes.
/// </summary>
/// <remarks>
/// <para>Testable at all because the reader is an interface. The real one needs ProteoWizard and a
/// multi-gigabyte instrument file; a fake needs neither, so this runs in CI on a build with no pwiz
/// at all - which is the build the cross-platform jobs use.</para>
///
/// <para>This is the piece that most needs it. It decides which replicates get measured and which
/// are taken from the cache, and both of its failure modes are silent: measuring nothing looks like
/// a small cohort, and reusing too much looks like a fast run.</para>
/// </remarks>
[Collection("IonAccountingRun")]
public class IonAccountingRunTests : IDisposable
{
    private readonly List<string> _dirs = new();

    public void Dispose()
    {
        Ms2SignalReaders.Clear();
        foreach (var dir in _dirs)
        {
            try
            {
                Directory.Delete(dir, recursive: true);
            }
            catch (IOException)
            {
                // A temp directory that will not delete is not a test failure.
            }
        }
    }

    /// <summary>
    /// A reader that invents a spectrum count and reports how many times it was asked to read, so a
    /// test can tell measuring from reusing.
    /// </summary>
    private sealed class FakeReader : IIonAccountingReader
    {
        private int _reads;

        public int Reads => Volatile.Read(ref _reads);

        /// <summary>Files this reader should fail on, to exercise the not-usable path.</summary>
        public HashSet<string> FailOn { get; } = new(StringComparer.OrdinalIgnoreCase);

        public string Describe() => "fake";

        public bool CanRead(string dataPath) =>
            dataPath.EndsWith(".mzML", StringComparison.OrdinalIgnoreCase);

        public Ms2SignalRecord Read(
            string dataPath, Action<string>? log = null, CancellationToken ct = default) =>
            Ms2SignalRecord.Unavailable(dataPath, Ms2ReadStatus.Failed, Describe(), "not implemented");

        public IonAccountingRecord ReadAccounting(
            string dataPath, IonAccountingRequest request, Action<string>? log = null,
            CancellationToken ct = default)
        {
            Interlocked.Increment(ref _reads);
            var name = Path.GetFileName(dataPath);
            if (FailOn.Contains(name))
            {
                return IonAccountingRecord.Unavailable(
                    dataPath, Ms2ReadStatus.Failed, Describe(), "asked to fail");
            }

            // Plausible magnitudes, so the ion-scale guard stays quiet: ~3.7e5 ions per MS1 scan.
            var cycles = new List<IonCycle>();
            for (var i = 0; i < 3; i++)
            {
                cycles.Add(new IonCycle(
                    i, i * 0.5, i * 0.5 + 0.5, 1, 10,
                    Ms1Acquired: 3.7e5, Ms2Acquired: 1.2e5,
                    Ms1Assigned: 1.5e5, Ms2Assigned: 4.0e3));
            }
            return new IonAccountingRecord(
                dataPath, Ms2ReadStatus.Ok, Describe(), 3, 30,
                Ms1Acquired: 3 * 3.7e5, Ms2Acquired: 3 * 1.2e5,
                Ms1Assigned: 3 * 1.5e5, Ms2Assigned: 3 * 4.0e3,
                Ms2Explained: 0, HasExplained: false,
                new double[request.ListCount], new double[request.ListCount],
                0, 1.5, 0, 0, cycles);
        }
    }

    [Fact]
    public void WithNoReaderInTheBuildNothingIsMeasuredAndTheReasonIsLogged()
    {
        Ms2SignalReaders.Clear();
        var dir = Seed(out var rawDir);
        var log = new List<string>();

        var result = Compute(dir, rawDir, log);

        Assert.Null(result);
        Assert.Contains(log, l => l.Contains("no instrument-file reader", StringComparison.Ordinal));
    }

    [Fact]
    public void AMissingMergedDataDirectoryIsReportedRatherThanThrowing()
    {
        Ms2SignalReaders.Register(new FakeReader());
        var dir = TempDir();
        Directory.CreateDirectory(Path.Combine(dir, "raw"));
        var log = new List<string>();

        var result = Compute(dir, Path.Combine(dir, "raw"), log);

        Assert.Null(result);
        Assert.Contains(log, l => l.Contains("merged_data", StringComparison.Ordinal));
    }

    [Fact]
    public void EveryPairedReplicateIsMeasuredAndCached()
    {
        var reader = new FakeReader();
        Ms2SignalReaders.Register(reader);
        var dir = Seed(out var rawDir);
        var log = new List<string>();

        var result = Compute(dir, rawDir, log);

        Assert.NotNull(result);
        Assert.Equal(2, reader.Reads);
        Assert.Equal(2, result!.Usable.Count);
        Assert.All(result.Usable, r => Assert.False(r.IonScaleImplausible));
        Assert.All(result.Usable, r => Assert.False(r.Exceeded));

        // 1.5e5 of 3.7e5 at MS1, 4.0e3 of 1.2e5 at MS2.
        Assert.Equal(1.5 / 3.7, result.Usable[0].Ms1Fraction, 6);
        Assert.Equal(4.0 / 120.0, result.Usable[0].Ms2Fraction, 6);

        // Written, and written per replicate rather than once at the end.
        Assert.True(File.Exists(Path.Combine(dir, IonAccountingStore.FileName)));
        Assert.Equal(6, IonAccountingStore.ReadCycles(dir).Count);   // 2 replicates x 3 cycles
    }

    /// <summary>
    /// The behavior that a <c>--max</c> spot check made necessary: a cache keyed for these settings
    /// is reused only for the replicates it actually covers, and the rest are measured.
    /// </summary>
    [Fact]
    public void ASecondRunMeasuresOnlyWhatTheCacheDoesNotCover()
    {
        var reader = new FakeReader();
        Ms2SignalReaders.Register(reader);
        var dir = Seed(out var rawDir);

        // First pass: one replicate only.
        var first = Compute(dir, rawDir, new List<string>(), maxReplicates: 1);
        Assert.NotNull(first);
        Assert.Equal(1, reader.Reads);
        Assert.Single(first!.Usable);

        // Second pass over the same settings: the covered replicate is reused, the other measured.
        var log = new List<string>();
        var second = Compute(dir, rawDir, log);

        Assert.NotNull(second);
        Assert.Equal(2, reader.Reads);            // ONE more read, not two
        Assert.Equal(2, second!.Usable.Count);
        Assert.Contains(log, l => l.Contains("covers 1 replicate", StringComparison.Ordinal));

        // The reused replicate kept its cycles: they are read before the first save rewrites them.
        Assert.Equal(6, IonAccountingStore.ReadCycles(dir).Count);

        // And a third pass with nothing missing reads nothing at all.
        var third = Compute(dir, rawDir, new List<string>());
        Assert.NotNull(third);
        Assert.Equal(2, reader.Reads);
        Assert.Equal(2, third!.Usable.Count);
    }

    /// <summary>
    /// A replicate whose file could not be read is NOT coverage - it is retried, not treated as done.
    /// </summary>
    [Fact]
    public void AFailedReplicateIsMeasuredAgainOnTheNextRun()
    {
        var reader = new FakeReader();
        reader.FailOn.Add("r2.mzML");
        Ms2SignalReaders.Register(reader);
        var dir = Seed(out var rawDir);

        var first = Compute(dir, rawDir, new List<string>());
        Assert.NotNull(first);
        Assert.Equal(2, reader.Reads);
        Assert.Single(first!.Usable);              // only r1 produced numbers

        reader.FailOn.Clear();
        var second = Compute(dir, rawDir, new List<string>());

        Assert.NotNull(second);
        Assert.Equal(3, reader.Reads);             // r2 retried, r1 reused
        Assert.Equal(2, second!.Usable.Count);
    }

    /// <summary>
    /// A replicate with no data file of its own gets a row with no numbers, so the plot shows a gap
    /// rather than omitting the injection or drawing a zero.
    /// </summary>
    [Fact]
    public void AReplicateWithNoDataFileStillGetsARow()
    {
        Ms2SignalReaders.Register(new FakeReader());
        var dir = Seed(out var rawDir, replicates: new[] { "r1", "r2", "r3" }, files: new[] { "r1", "r2" });
        var log = new List<string>();

        var result = Compute(dir, rawDir, log);

        Assert.NotNull(result);
        Assert.Equal(3, result!.Rows.Count);
        Assert.Equal(2, result.Usable.Count);

        var gap = result.Rows.Single(r => r.Sample.StartsWith("r3", StringComparison.Ordinal));
        Assert.False(gap.IsUsable);
        Assert.Equal("", gap.DataFile);
        Assert.Contains(log, l => l.Contains("matched no data file", StringComparison.Ordinal));
    }

    // ------------------------------------------------------------------ fixture

    private static IonAccountingResult? Compute(
        string dir, string rawDir, List<string> log, int maxReplicates = 0) =>
        IonAccountingRun.Compute(
            dir, rawDir,
            IsolationScheme.Cycle("c", start: 400, step: 4, width: 4, count: 2),
            ProductMassTolerance.ParseSetting("10 ppm")!,
            ProductMassTolerance.ParseSetting("10 ppm"),
            Array.Empty<ProteinList>(),
            sampleTypes: null, log: log.Add, maxReplicates: maxReplicates);

    /// <summary>
    /// An output directory with the three things Compute reads: a merged table, the peptide matrix
    /// whose row set is the assigned set, and a directory of data files to pair against.
    /// </summary>
    private string Seed(
        out string rawDir, string[]? replicates = null, string[]? files = null)
    {
        replicates ??= new[] { "r1", "r2" };
        files ??= replicates;

        var dir = TempDir();
        rawDir = Path.Combine(dir, "raw");
        Directory.CreateDirectory(rawDir);
        foreach (var file in files)
            File.WriteAllText(Path.Combine(rawDir, file + ".mzML"), "");

        // merged_data as a single parquet, which MergedDataset.Open accepts.
        var merged = Path.Combine(dir, "merged_data");
        Directory.CreateDirectory(merged);
        WriteMerged(Path.Combine(merged, "_pep_bucket=0"), replicates);

        // The peptide matrix: its ROW SET is the assigned set. One column of keys is enough.
        ParquetWideWriterKeys(
            Path.Combine(dir, "peptides_rollup.parquet"), new[] { "PEPTIDEK", "OTHERPEPK" });

        // sample_metadata.csv is where the replicate list comes from, keyed on sample_id - the
        // merged table's "<replicate>__@__<batch>" key rather than the bare name.
        var metadata = new List<string> { "sample_id,sample,sample_type,batch" };
        foreach (var replicate in replicates)
            metadata.Add($"{replicate}__@__batch,{replicate},experimental,batch");
        File.WriteAllLines(Path.Combine(dir, "sample_metadata.csv"), metadata);

        return dir;
    }

    private static void WriteMerged(string bucketDir, IReadOnlyList<string> replicates)
    {
        Directory.CreateDirectory(bucketDir);
        var rows = new List<string>();
        foreach (var replicate in replicates)
        {
            foreach (var (peptide, fragment, mz) in new[]
                     {
                         ("PEPTIDEK", "precursor", 402.0),
                         ("PEPTIDEK", "y5", 600.0),
                         ("OTHERPEPK", "y4", 500.0),
                     })
            {
                rows.Add("SELECT "
                    + $"'{replicate}__@__batch' AS \"Sample ID\", "
                    + $"'{peptide}' AS \"Peptide Modified Sequence\", "
                    + $"'{fragment}' AS \"Fragment Ion\", "
                    + "1.0 AS \"Area\", "
                    + $"{Num(402.0)} AS \"Precursor Mz\", "
                    + $"{Num(mz)} AS \"Product Mz\", "
                    + "0.0 AS \"Start Time\", "
                    + "2.0 AS \"End Time\"");
            }
        }

        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        var target = Path.Combine(bucketDir, "data_0.parquet").Replace('\\', '/');
        cmd.CommandText = $"COPY ({string.Join(" UNION ALL ", rows)}) TO '{target}' (FORMAT PARQUET)";
        cmd.ExecuteNonQuery();
    }

    private static void ParquetWideWriterKeys(string path, IReadOnlyList<string> peptides)
    {
        SkylinePrism.Core.IO.ParquetWideWriter.Write(
            path,
            new[] { SkylinePrism.Core.IO.ParquetWideWriter.Strings("peptide", peptides.ToArray()) },
            Array.Empty<string>(), Array.Empty<double[]>(), peptides.Count);
    }

    private static string Num(double value) => value.ToString("R", CultureInfo.InvariantCulture);

    private string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_ionrun_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _dirs.Add(dir);
        return dir;
    }
}
