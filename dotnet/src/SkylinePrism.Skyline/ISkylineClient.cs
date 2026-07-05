#nullable enable

using System;
using System.Collections.Generic;

namespace SkylinePrism.Skyline;

/// <summary>Inline rows read from a Skyline report/document-grid view (column headers + string cells).</summary>
public sealed record ReportRows(IReadOnlyList<string> Columns, IReadOnlyList<string[]> Rows);

/// <summary>
/// The subset of Skyline's JSON-RPC client that <see cref="SkylineReportDriver"/> needs. Extracting
/// it as an interface lets the driver be tested with a fake instead of a live Skyline pipe.
/// </summary>
public interface ISkylineClient
{
    string GetDocumentPath();
    string GetVersion();
    void ExportReport(string reportName, string filePath, string culture);
    string[] GetSettingsListNames(string listType, string? groupName);
    void RunCommandSilent(string[] args);

    /// <summary>Column names of the Replicate entity (built-ins + user-defined document annotations).</summary>
    string[] GetReplicateColumns();

    /// <summary>Read a per-replicate report (one row per replicate) for the given columns; null if unavailable.</summary>
    ReportRows? GetReplicateReport(IReadOnlyList<string> selectColumns);
}

/// <summary>
/// Opens a Skyline connection per call and hands the driver an <see cref="ISkylineClient"/>.
/// <see cref="SkylineSession"/> is the production implementation (a JSON-RPC pipe); tests supply a fake.
/// </summary>
public interface ISkylineExecutor
{
    T Execute<T>(Func<ISkylineClient, T> action);
    void Execute(Action<ISkylineClient> action);
}
