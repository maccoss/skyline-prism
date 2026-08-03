#nullable enable

namespace SkylinePrism.Skyline;

/// <summary>
/// The report files exported for ONE Skyline document, whatever produced them
/// (<see cref="SkylineReportDriver"/> over the RPC, or <see cref="HeadlessSkylineExporter"/> via
/// SkylineCmd).
/// </summary>
/// <param name="InputPath">
/// The PRISM transition report the pipeline reads. Its file stem is <paramref name="DocumentLabel"/>,
/// because <c>DuckDbMerge</c> derives each input's Batch / Source Document label from the file stem -
/// that is what keeps replicates from different documents distinct in the merged
/// "&lt;replicate&gt;__@__&lt;document&gt;" sample IDs.
/// </param>
/// <param name="InputIsParquet">
/// True for the RPC path (parquet, ~20x faster to read). Headless SkylineCmd export only supports
/// <c>--report-format=csv|tsv</c>, so a closed document always yields CSV.
/// </param>
/// <param name="ReplicatesCsv">The per-replicate metadata CSV, or null when it could not be produced.</param>
/// <param name="DocumentPath">The .sky path, when known.</param>
/// <param name="DocumentLabel">The batch / source-document label for this input (the report file stem).</param>
public sealed record ExportedReports(
    string InputPath,
    bool InputIsParquet,
    string? ReplicatesCsv,
    string? DocumentPath,
    string DocumentLabel);
