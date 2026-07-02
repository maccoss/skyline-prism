#nullable enable

using System;

namespace SkylinePrism.Skyline;

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
