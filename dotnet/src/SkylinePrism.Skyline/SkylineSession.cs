#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Text.Json;
using SkylineTool;

namespace SkylinePrism.Skyline;

/// <summary>
/// Connection factory for Skyline's JSON-RPC pipe. Skyline closes the pipe after each
/// request/response (the SkylineMcpServer.Invoke connect-per-call pattern), so this class
/// does NOT hold a pipe open - it remembers how to connect and opens a fresh pipe inside
/// each <see cref="Execute"/> call. Ported from skyline-cadenza's SkylineSession.
/// </summary>
public sealed class SkylineSession : ISkylineExecutor
{
    public string PipeName { get; }
    public int? SkylineProcessId { get; }
    public TimeSpan ConnectTimeout { get; }

    private SkylineSession(string pipeName, int? processId, TimeSpan timeout)
    {
        PipeName = pipeName;
        SkylineProcessId = processId;
        ConnectTimeout = timeout;
    }

    /// <summary>
    /// Construct a session from <c>args[0]</c> (the <c>$(SkylineConnection)</c> pipe name
    /// Skyline passes external tools). Falls back to discovering any running instance via
    /// <c>~/.skyline-mcp/</c> when no argument is supplied.
    /// </summary>
    public static SkylineSession FromArguments(string[] args, TimeSpan? timeout = null)
    {
        var to = timeout ?? TimeSpan.FromSeconds(5);
        if (args.Length > 0 && !string.IsNullOrWhiteSpace(args[0]))
        {
            // Skyline expands $(SkylineConnection) to the LEGACY ToolService pipe name; the
            // JSON-RPC server listens on a derived name. Transform via GetJsonPipeName.
            var raw = args[0];
            var jsonName = raw.StartsWith(JsonToolConstants.JSON_PIPE_PREFIX, StringComparison.Ordinal)
                ? raw
                : JsonToolConstants.GetJsonPipeName(raw);
            return new SkylineSession(jsonName, null, to);
        }

        var info = DiscoverMostRecent()
            ?? throw new InvalidOperationException(
                "No Skyline pipe-name argument was passed and no running Skyline instance was found in ~/.skyline-mcp/.");
        return new SkylineSession(info.PipeName, info.ProcessId, to);
    }

    /// <summary>Open a fresh pipe, hand the client to <paramref name="action"/>, and dispose.</summary>
    public T Execute<T>(Func<ISkylineClient, T> action)
    {
        using var pipe = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut);
        pipe.Connect((int)ConnectTimeout.TotalMilliseconds);
        pipe.ReadMode = PipeTransmissionMode.Message;
        return action(new JsonClientAdapter(new SkylineJsonToolClient(pipe)));
    }

    public void Execute(Action<ISkylineClient> action)
    {
        using var pipe = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut);
        pipe.Connect((int)ConnectTimeout.TotalMilliseconds);
        pipe.ReadMode = PipeTransmissionMode.Message;
        action(new JsonClientAdapter(new SkylineJsonToolClient(pipe)));
    }

    /// <summary>Forwards <see cref="ISkylineClient"/> calls to the concrete JSON-RPC client.</summary>
    private sealed class JsonClientAdapter : ISkylineClient
    {
        private readonly SkylineJsonToolClient _c;
        public JsonClientAdapter(SkylineJsonToolClient c) => _c = c;
        public string GetDocumentPath() => _c.GetDocumentPath();
        public string GetVersion() => _c.GetVersion();
        public void ExportReport(string reportName, string filePath, string culture) => _c.ExportReport(reportName, filePath, culture);
        public string[] GetSettingsListNames(string listType, string? groupName) => _c.GetSettingsListNames(listType, groupName);
        public string[] GetSettingsListSelectedItems(string listType) => _c.GetSettingsListSelectedItems(listType);
        public string GetSettingsListItem(string listType, string itemName) => _c.GetSettingsListItem(listType, itemName);
        public void RunCommandSilent(string[] args) => _c.RunCommandSilent(args);

        public string[] GetReplicateColumns()
        {
            // The built-in "Replicates" view is not a named report; enumerate the Replicate entity's
            // columns (built-ins + user-defined document annotations) from its doc topic instead.
            var detail = _c.GetReportDocTopic("Replicate", "document_grid");
            return detail?.Columns?.Select(col => col.Name).Where(n => !string.IsNullOrWhiteSpace(n)).ToArray()
                   ?? Array.Empty<string>();
        }

        public ReportRows? GetReplicateReport(IReadOnlyList<string> selectColumns)
        {
            var def = new ReportDefinition
            {
                Select = selectColumns.ToArray(),
                PivotReplicate = false, // one row per replicate
                DataSource = "document_grid",
            };
            // count=0 returns the shape (TotalRows); then pull every row.
            var shape = _c.GetReportFromDefinitionRows(def, 0, 0, false, "invariant");
            var total = shape?.TotalRows ?? 0;
            var res = total > 0
                ? _c.GetReportFromDefinitionRows(def, 0, total, false, "invariant")
                : shape;
            if (res?.Columns is null || res.Rows is null)
                return null;
            return new ReportRows(res.Columns.Select(c => c.Name).ToArray(), res.Rows);
        }
    }

    public static List<ConnectionInfo> DiscoverAll()
    {
        var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".skyline-mcp");
        if (!Directory.Exists(dir))
            return new List<ConnectionInfo>();

        var result = new List<ConnectionInfo>();
        foreach (var file in Directory.EnumerateFiles(dir, "connection-*.json"))
        {
            try
            {
                using var fs = File.OpenRead(file);
                using var doc = JsonDocument.Parse(fs);
                var root = doc.RootElement;
                if (!root.TryGetProperty("pipe_name", out var pipeProp))
                    continue;
                var pipeName = pipeProp.GetString();
                if (string.IsNullOrWhiteSpace(pipeName))
                    continue;
                int? pid = root.TryGetProperty("process_id", out var pidProp) ? pidProp.GetInt32() : null;
                DateTime? connectedAt = null;
                if (root.TryGetProperty("connected_at", out var atProp) &&
                    DateTime.TryParse(atProp.GetString(), out var parsed))
                {
                    connectedAt = parsed;
                }
                result.Add(new ConnectionInfo
                {
                    PipeName = pipeName!,
                    ProcessId = pid,
                    ConnectedAt = connectedAt,
                    SkylineVersion = root.TryGetProperty("skyline_version", out var v) ? v.GetString() : null,
                });
            }
            catch
            {
                // Ignore unreadable / stale connection files.
            }
        }
        return result;
    }

    public static ConnectionInfo? DiscoverMostRecent()
        => DiscoverAll().OrderByDescending(c => c.ConnectedAt ?? DateTime.MinValue).FirstOrDefault();

    public sealed class ConnectionInfo
    {
        public required string PipeName { get; init; }
        public int? ProcessId { get; init; }
        public DateTime? ConnectedAt { get; init; }
        public string? SkylineVersion { get; init; }
    }
}
