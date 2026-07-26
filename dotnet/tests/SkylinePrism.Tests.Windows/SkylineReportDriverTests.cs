using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Drives SkylineReportDriver against a fake ISkylineExecutor (no live Skyline): the parquet-first /
/// CSV-fallback export decision, metadata report-name resolution, report-list dedup, and .blib
/// discovery - the report-driver logic that historically had the most bugs.
/// </summary>
public class SkylineReportDriverTests
{
    private sealed class FakeClient : ISkylineClient
    {
        public string DocumentPath = "";
        public string Version = "24.1";
        public readonly Dictionary<string, List<string>> ReportsByGroup = new();
        public readonly List<(string Report, string Path)> Exports = new();
        public readonly List<string[]> Commands = new();
        public readonly HashSet<string> ThrowForReports = new(StringComparer.OrdinalIgnoreCase);
        public Action<string, string>? OnExport; // (report, path) -> write the file

        public string GetDocumentPath() => DocumentPath;
        public string GetVersion() => Version;

        public void ExportReport(string reportName, string filePath, string culture)
        {
            Exports.Add((reportName, filePath));
            if (ThrowForReports.Contains(reportName))
                throw new InvalidOperationException($"no such report '{reportName}'");
            OnExport?.Invoke(reportName, filePath);
        }

        public string[] GetSettingsListNames(string listType, string? groupName)
            => ReportsByGroup.TryGetValue(groupName ?? "", out var l) ? l.ToArray() : Array.Empty<string>();

        // Settings-list items keyed by "<listType>/<itemName>"; SelectedByList holds the active items.
        public readonly Dictionary<string, string> SettingsItems = new(StringComparer.Ordinal);
        public readonly Dictionary<string, string[]> SelectedByList = new(StringComparer.Ordinal);

        public string[] GetSettingsListSelectedItems(string listType)
            => SelectedByList.TryGetValue(listType, out var s) ? s : Array.Empty<string>();

        public string GetSettingsListItem(string listType, string itemName)
            => SettingsItems.TryGetValue($"{listType}/{itemName}", out var xml) ? xml : "";

        public void RunCommandSilent(string[] args) => Commands.Add(args);

        // Replicate-grid read: no report by default (so the driver falls back to a saved report).
        public string[] ReplicateColumns = Array.Empty<string>();
        public ReportRows? ReplicateReport;
        public IReadOnlyList<string>? RequestedSelect;

        public string[] GetReplicateColumns() => ReplicateColumns;

        public ReportRows? GetReplicateReport(IReadOnlyList<string> selectColumns)
        {
            RequestedSelect = selectColumns;
            return ReplicateReport;
        }
    }

    private sealed class FakeExecutor : ISkylineExecutor
    {
        public readonly FakeClient Client;
        public FakeExecutor(FakeClient c) => Client = c;
        public T Execute<T>(Func<ISkylineClient, T> action) => action(Client);
        public void Execute(Action<ISkylineClient> action) => action(Client);
    }

    private static string TempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "prism_rpc_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void WriteValidParquet(string path)
    {
        var b = new byte[16];
        "PAR1"u8.CopyTo(b);
        "PAR1"u8.CopyTo(b.AsSpan(12)); // head + tail magic
        File.WriteAllBytes(path, b);
    }

    [Fact]
    public void Export_PrefersParquet_WhenValid()
    {
        var client = new FakeClient
        {
            DocumentPath = Path.Combine(TempDir(), "doc.sky"),
            OnExport = (report, path) =>
            {
                if (path.EndsWith(".parquet")) WriteValidParquet(path);
                else File.WriteAllText(path, "a,b\n1,2\n");
            },
        };
        client.ReportsByGroup[""] = new List<string> { "PRISM-Replicates" };
        var work = TempDir();

        var res = new SkylineReportDriver(new FakeExecutor(client)).Export(work);

        Assert.True(res.InputIsParquet);
        Assert.EndsWith("PRISM.parquet", res.InputPath);
        Assert.True(File.Exists(res.InputPath));
        Assert.NotNull(res.ReplicatesCsv); // metadata report exported too
    }

    [Fact]
    public void Export_FallsBackToCsv_WhenParquetInvalid()
    {
        var client = new FakeClient
        {
            OnExport = (_, path) => File.WriteAllText(path, "a,b\n1,2\n"), // parquet path gets non-parquet bytes
        };
        var work = TempDir();

        var res = new SkylineReportDriver(new FakeExecutor(client)).Export(work);

        Assert.False(res.InputIsParquet);
        Assert.EndsWith("PRISM.csv", res.InputPath);
        Assert.True(File.Exists(res.InputPath));
        Assert.False(File.Exists(Path.Combine(work, "PRISM.parquet"))); // invalid parquet cleaned up
    }

    [Fact]
    public void ExportMetadata_ResolvesRequestedNameCaseInsensitively()
    {
        var client = new FakeClient { OnExport = (_, p) => File.WriteAllText(p, "x") };
        client.ReportsByGroup[""] = new List<string> { "MyReplicates" };
        var work = TempDir();

        new SkylineReportDriver(new FakeExecutor(client)).Export(work, metadataReportName: "myreplicates");

        // The available casing wins, and it is what gets exported to Metadata.csv.
        Assert.Contains(client.Exports, e => e.Report == "MyReplicates" && e.Path.EndsWith("Metadata.csv"));
    }

    [Fact]
    public void ExportMetadata_DefaultsToPrismReplicates_WhenNoneRequested()
    {
        var client = new FakeClient { OnExport = (_, p) => File.WriteAllText(p, "x") };
        var work = TempDir();

        new SkylineReportDriver(new FakeExecutor(client)).Export(work);

        Assert.Contains(client.Exports, e => e.Report == "PRISM-Replicates" && e.Path.EndsWith("Metadata.csv"));
    }

    [Fact]
    public void ExportMetadata_ReadsReplicatesGrid_WhenAvailable()
    {
        var client = new FakeClient
        {
            OnExport = (_, p) => File.WriteAllText(p, "x"),
            // Built-ins + two user-defined annotations (Condition, Subject); FilePath is built-in noise.
            ReplicateColumns = new[] { "Replicate", "ReplicateName", "FilePath", "SampleType", "BatchName", "Condition", "Subject" },
            ReplicateReport = new ReportRows(
                new[] { "ReplicateName", "SampleType", "Condition" },
                new List<string[]>
                {
                    new[] { "R1", "Standard", "A, treated" }, // comma forces CSV quoting
                    new[] { "R2", "Quality Control", "B" },
                }),
        };
        var work = TempDir();

        new SkylineReportDriver(new FakeExecutor(client)).Export(work);

        // The select carries the Replicates view's standard columns + annotations, but not file/instrument
        // noise (FilePath) nor Replicate-entity properties the built-in view doesn't show (BatchName,
        // SampleDilutionFactor, SampleId).
        Assert.NotNull(client.RequestedSelect);
        Assert.Contains("SampleType", client.RequestedSelect!);
        Assert.Contains("Condition", client.RequestedSelect!);
        Assert.Contains("Subject", client.RequestedSelect!);
        Assert.DoesNotContain("FilePath", client.RequestedSelect!);
        Assert.DoesNotContain("BatchName", client.RequestedSelect!);
        Assert.DoesNotContain("SampleDilutionFactor", client.RequestedSelect!);
        Assert.DoesNotContain("SampleId", client.RequestedSelect!);

        // Metadata.csv is written from the grid, not exported from a saved report.
        Assert.DoesNotContain(client.Exports, e => e.Path.EndsWith("Metadata.csv"));
        var csv = File.ReadAllText(Path.Combine(work, "Metadata.csv"));
        Assert.StartsWith("ReplicateName,SampleType,Condition", csv);
        Assert.Contains("R1,Standard,\"A, treated\"", csv);
        Assert.Contains("R2,Quality Control,B", csv);
    }

    [Fact]
    public void ExportMetadata_ReturnsNull_WhenReportExportThrows()
    {
        var client = new FakeClient { OnExport = (_, p) => File.WriteAllText(p, "x") };
        client.ThrowForReports.Add("PRISM-Replicates");
        var work = TempDir();

        var res = new SkylineReportDriver(new FakeExecutor(client)).Export(work);

        Assert.Null(res.ReplicatesCsv);
    }

    [Fact]
    public void ListAvailableReports_DedupsAcrossGroups_AndDropsBlanks()
    {
        var client = new FakeClient();
        client.ReportsByGroup[""] = new List<string> { "A", "B" };
        client.ReportsByGroup["main"] = new List<string> { "B", "C" };
        client.ReportsByGroup["external_tools"] = new List<string> { "C", "D", "" };

        var names = new SkylineReportDriver(new FakeExecutor(client)).ListAvailableReports();

        Assert.Equal(new[] { "A", "B", "C", "D" }, names.OrderBy(x => x, StringComparer.Ordinal));
    }

    [Fact]
    public void ListDocumentLibraries_ReturnsBlibsNextToDocument_DocNamedFirst()
    {
        var dir = TempDir();
        File.WriteAllText(Path.Combine(dir, "doc.blib"), "x");
        File.WriteAllText(Path.Combine(dir, "aux.blib"), "x");
        File.WriteAllText(Path.Combine(dir, "notes.txt"), "x");
        var client = new FakeClient { DocumentPath = Path.Combine(dir, "doc.sky") };

        var libs = new SkylineReportDriver(new FakeExecutor(client)).ListDocumentLibraries();

        Assert.Equal(2, libs.Count);
        Assert.EndsWith("doc.blib", libs[0]); // document-named library listed first
    }

    [Fact]
    public void Export_WithBatchAnnotation_InstallsDynamicReplicatesReport()
    {
        var client = new FakeClient { OnExport = (_, p) => File.WriteAllText(p, "x") };
        var work = TempDir();

        new SkylineReportDriver(new FakeExecutor(client)).Export(work, batchAnnotation: "Batch");

        // The dynamic PRISM-Replicates install issues a --report-add command.
        Assert.Contains(client.Commands, cmd => cmd.Any(a => a.StartsWith("--report-add=")));
    }

    [Fact]
    public void GetDigestionEnzyme_MapsTrypsinFromDocument()
    {
        var client = new FakeClient();
        client.SelectedByList["Enzymes"] = new[] { "Trypsin" };
        client.SettingsItems["Enzymes/Trypsin"] = "<enzyme name=\"Trypsin\" cut=\"KR\" no_cut=\"P\" sense=\"C\" />";

        var enzyme = new SkylineReportDriver(new FakeExecutor(client)).GetDigestionEnzyme();

        Assert.Equal("trypsin", enzyme);
    }

    [Fact]
    public void GetDigestionEnzyme_DistinguishesTrypsinP()
    {
        var client = new FakeClient();
        client.SelectedByList["Enzymes"] = new[] { "Trypsin/P" };
        client.SettingsItems["Enzymes/Trypsin/P"] = "<enzyme name=\"Trypsin/P\" cut=\"KR\" no_cut=\"\" sense=\"C\" />";

        var enzyme = new SkylineReportDriver(new FakeExecutor(client)).GetDigestionEnzyme();

        Assert.Equal("trypsin/p", enzyme);
    }

    [Fact]
    public void GetDigestionEnzyme_ReturnsNull_WhenNoEnzymeSelected()
    {
        var client = new FakeClient(); // no selected "Enzymes" item
        Assert.Null(new SkylineReportDriver(new FakeExecutor(client)).GetDigestionEnzyme());
    }

    [Fact]
    public void GetDigestionEnzyme_ReturnsNull_ForUnmappableEnzyme()
    {
        var client = new FakeClient();
        client.SelectedByList["Enzymes"] = new[] { "CNBr" };
        client.SettingsItems["Enzymes/CNBr"] = "<enzyme name=\"CNBr\" cut=\"M\" no_cut=\"\" sense=\"C\" />";

        // M (methionine) has no PRISM enzyme equivalent -> null so the caller keeps the config default.
        Assert.Null(new SkylineReportDriver(new FakeExecutor(client)).GetDigestionEnzyme());
    }
}
