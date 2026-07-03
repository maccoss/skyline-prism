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

    /// <summary>Read a report/document-grid view's content inline (all columns); null if unavailable.</summary>
    ReportRows? GetReportRows(string reportName);

    /// <summary>Names of the document-grid views (used to locate the built-in "Replicates" view).</summary>
    string[] ListDocumentGridViews();
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
