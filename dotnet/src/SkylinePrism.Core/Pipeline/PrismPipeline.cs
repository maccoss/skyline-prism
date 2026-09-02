using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using DuckDB.NET.Data;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Parsimony;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Rollup;
using System.Threading;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// End-to-end PRISM pipeline (cmd_run), orchestrating Stage 1..5: merge -> transition rollup
/// -> peptide normalization -> peptide ComBat -> parsimony -> protein rollup -> protein
/// normalization -> protein ComBat -> outputs. Produces the same output files as the Python
/// pipeline. QC report generation (Stage 5b) is Layer 8.
/// </summary>
public sealed class PrismPipeline
{
    public sealed record Result(
        int NPeptides, int NProteins, int NSamples, IReadOnlyList<string> Batches);

    /// <summary>Compact elapsed: "3m 12s" or "4.8s".</summary>
    private static string Fmt(TimeSpan t) =>
        t.TotalMinutes >= 1 ? $"{(int)t.TotalMinutes}m {t.Seconds:00}s" : $"{t.TotalSeconds:n1}s";

    /// <summary>
    /// Run the pipeline. <paramref name="cancellationToken"/> is honoured at every stage boundary and
    /// inside the long-running stages (the merge query, the transition rollup's producer, and the
    /// row-group loops of Stage 2b/2c), so a Stop takes effect in seconds rather than at the end of
    /// whatever stage happens to be running. Outputs written before the stop are left in place - they
    /// are intermediates of an incomplete run, not results.
    /// </summary>
    public static Result Run(
        IReadOnlyList<string> inputs, string outputDir, PrismConfig config,
        IReadOnlyList<string>? metadataPaths = null, Action<string>? log = null,
        bool forceReprocess = false, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(outputDir);
        var reportRaw = log ?? (_ => { });

        // Per-stage timing in the run log. Added because every performance conclusion about this
        // pipeline so far came from an external sampler wrapped around the process: the log said what
        // each stage DID but never what it COST, so a stage that doubled in time was invisible until
        // someone happened to wall-clock the whole run. A stage that is slow on a user's cohort should
        // be visible in the artifact they already send us.
        var stageTimer = Stopwatch.StartNew();
        var totalTimer = Stopwatch.StartNew();
        var timings = new List<(string Stage, TimeSpan Elapsed)>();
        var currentStage = "";

        void EndStage()
        {
            if (currentStage.Length == 0)
                return;
            timings.Add((currentStage, stageTimer.Elapsed));
            reportRaw($"  [{currentStage} took {Fmt(stageTimer.Elapsed)}]");
        }

        // Stage banners are the natural boundary, so timing hangs off them rather than off a parallel
        // set of markers that could drift out of step with the stages themselves.
        void report(string line)
        {
            if (line.StartsWith("Stage ", StringComparison.Ordinal))
            {
                EndStage();
                currentStage = line.Split(':')[0];
                stageTimer.Restart();
            }
            reportRaw(line);
        }

        // Per-stage reuse for a re-run into this directory. A stage whose inputs and settings are
        // unchanged keeps its output; --force-reprocess reuses nothing but still records, so the next
        // run benefits. See StageDependencies for what "unchanged" means, key by key.
        var stageCache = StageCache.Load(outputDir, forceReprocess);

        report("============================================================");
        report("Stage 1: Merge / prepare input");
        report("============================================================");
        var mergedPath = Path.Combine(outputDir, "merged_data");
        var cachePath = mergedPath + ".cache.json";
        // The input files AND the column overrides: DuckDbMerge is given data.sample_column, and the
        // partitioning follows the peptide column, so a changed data.* override makes the merged table
        // wrong even though every input file is untouched. Fingerprinting only the files meant the
        // merge was reused while everything below it recomputed - from a stale table.
        var fingerprint = SourceFingerprint.Compute(inputs)
            + "|" + StageDependencies.Values(StageDependencies.Merge, config);

        DuckDbMerge.MergeResult merge;
        var cached = forceReprocess ? null : SourceFingerprint.TryRead(cachePath);
        if (cached is not null && cached.Fingerprint == fingerprint && MergedDataset.Exists(mergedPath))
        {
            // CacheEntry.SortColumn keeps its old name to stay readable by sidecars written before the
            // merge stopped sorting; what it holds is the PEPTIDE column (the partition key). Nothing
            // about a cached merge is sorted.
            merge = new DuckDbMerge.MergeResult(mergedPath, cached.SortColumn, cached.TotalRows);
            report($"  Reusing cached merge ({merge.TotalRows:N0} rows; inputs unchanged - "
                + "pass --force-reprocess to rebuild).");
        }
        else
        {
            merge = DuckDbMerge.Merge(
                inputs, mergedPath, replicateColumn: config.Data.SampleColumn,
                memoryBudgetMb: config.Processing.MergeMemoryMb,
                cancellationToken: cancellationToken);
            SourceFingerprint.Write(cachePath,
                new SourceFingerprint.CacheEntry(fingerprint, merge.TotalRows, merge.PeptideColumn));
            report($"  Merged {inputs.Count} report(s) -> {merge.TotalRows:N0} transition rows "
                + $"across {merge.Partitions} peptide partition(s).");
            // Worth naming: it is the first thing to look at when a stage is slow or fills a disk,
            // and it is not always beside the output.
            report($"  Scratch: {merge.TempDirectory} "
                + $"(override with the {DuckDbMerge.TempDirEnvVar} environment variable).");
            // Named because the automatic value is set partly from FREE memory: the same machine can
            // pick a different budget on two runs, and that is the explanation when one of them
            // spills and the other did not. It bounds the rollup's reader too, not just the merge.
            report($"  Memory budget: {merge.MemoryBudgetMb:N0} MB"
                + (config.Processing.MergeMemoryMb > 0
                    ? " (from processing.merge_memory_mb)."
                    : " (auto; set processing.merge_memory_mb to override)."));
        }

        // Schema-only read: never materialize the (potentially huge, 200-report) merged table
        // just to detect column names.
        var dataset = merge.Dataset();
        var cols = SkylineColumns.Detect(
            ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToHashSet(),
            config.Data.ToOverrides());
        var samples = MergedParquetReader.GetSortedSamples(
            dataset, cols.Sample, config.Processing.MergeMemoryMb);
        report($"  Columns: peptide='{cols.Peptide}', sample='{cols.Sample}', abundance='{cols.Abundance}'.");
        report($"  Samples: {samples.Count}.");

        // Resolve per-sample batch and type: prefer the Replicates metadata (Batch annotation /
        // Skyline Sample Type), else fall back to the Source Document batch + name patterns.
        var sourceBatchMap = GetBatchMap(dataset, cols);
        // Metadata files are matched to inputs positionally so each file's rows can be qualified by its
        // source document ("<replicate>__@__<document>"). Without that, a replicate name reused across
        // documents (every plate has a "Ref_01") would take the LAST file's type/batch for every document.
        // Only 1:1 metadata-per-input is unambiguous; otherwise fall back to bare replicate keys.
        var documentLabels = metadataPaths is not null && metadataPaths.Count == inputs.Count && inputs.Count > 1
            ? inputs.Select(p => Path.GetFileNameWithoutExtension(p) ?? p).ToList()
            : null;
        var metadata = ReplicateMetadata.TryLoad(
            metadataPaths, report,
            config.Data.SampleTypeColumn ?? config.Metadata.SampleTypeColumn,
            config.Data.BatchColumn ?? config.Metadata.BatchColumn,
            documentLabels);
        var resolvedBatch = new Dictionary<string, string>(StringComparer.Ordinal);
        var resolvedType = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var s in samples)
        {
            var rep = SampleIdToReplicate(s);
            resolvedBatch[s] = metadata?.BatchFor(s, rep)
                ?? sourceBatchMap.GetValueOrDefault(s, "batch1");
            resolvedType[s] = metadata?.TypeFor(s, rep)
                ?? ClassifySampleType(s, rep, config);
        }

        // If neither metadata nor Source Document distinguishes batches (all one label), fall back
        // to estimating batches from acquisition-time gaps (ports cli.py:estimate_batches_from_parquet).
        var estMethod = (config.BatchEstimation.Method ?? "auto").ToLowerInvariant();
        if (resolvedBatch.Values.Distinct().Count() <= 1
            && estMethod is not ("none" or "source")
            && cols.AcquiredTime is not null)
        {
            var est = BatchEstimator.Estimate(
                dataset, cols.Sample, cols.AcquiredTime, estMethod,
                config.BatchEstimation.NBatches, config.BatchEstimation.GapIqrMultiplier, report);
            var nEst = est.Values.Distinct().Count();
            if (nEst > 1)
            {
                foreach (var s in samples)
                {
                    var rep = SampleIdToReplicate(s);
                    var hasMeta = metadata?.HasBatchFor(s, rep) == true;
                    if (!hasMeta && est.TryGetValue(s, out var b))
                        resolvedBatch[s] = b;
                }
                report($"  Batch estimation: {nEst} batches from acquisition-time gaps "
                    + $"(method={estMethod}).");
            }
        }

        var batchLabels = samples.Select(s => resolvedBatch[s]).ToList();
        var batches = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
        var multiBatch = batches.Count >= 2;
        var peptideCombat = config.BatchCorrection.Enabled && config.BatchCorrection.PeptideLevel && multiBatch;
        var proteinCombat = config.BatchCorrection.Enabled && config.BatchCorrection.ProteinLevel && multiBatch;
        var combatNote = !multiBatch ? "skipped (needs >= 2 batches)"
            : !config.BatchCorrection.Enabled ? "disabled"
            : $"peptide={(peptideCombat ? "on" : "off")}, protein={(proteinCombat ? "on" : "off")}";
        // Which source supplied the batch labels (metadata Batch column > per-file Source Document >
        // acquisition-time estimation > a single default label).
        var metaProvidedBatch = metadata is not null
            && samples.Any(s => metadata.HasBatchFor(s, SampleIdToReplicate(s)));
        var batchSource = metaProvidedBatch ? "metadata report (Batch column)"
            : sourceBatchMap.Values.Distinct().Count() > 1 ? "Source Document (per-file)"
            : batches.Count > 1 ? "acquisition-time estimation"
            : "single label (no batch annotation)";
        report($"Batches: {batches.Count} from {batchSource} ({string.Join(", ", batches)}); ComBat {combatNote}.");

        // Stage 2: transition -> peptide.
        report("============================================================");
        cancellationToken.ThrowIfCancellationRequested();
        report($"Stage 2: Transition -> peptide rollup ({config.TransitionRollup.Method})");
        report("============================================================");
        var peptidesRollupPath = Path.Combine(outputDir, "peptides_rollup.parquet");
        var transitionCfg = new TransitionRollupConfig
        {
            Method = config.TransitionRollup.Method switch
            {
                "median_polish" => TransitionRollupMethod.MedianPolish,
                "topn" => TransitionRollupMethod.TopN,
                "consensus" => TransitionRollupMethod.Consensus,
                "library_assist" or "library-assisted" or "library_assisted" => TransitionRollupMethod.LibraryAssist,
                _ => TransitionRollupMethod.Sum,
            },
            MinTransitions = config.TransitionRollup.MinTransitions,
            TopNCount = config.TransitionRollup.TopnCount,
            TopNSelection = config.TransitionRollup.TopnSelection,
            TopNWeighting = config.TransitionRollup.TopnWeighting,
            ConsensusRegularization = config.TransitionRollup.ConsensusRegularization,
            UseMs1 = config.TransitionRollup.UseMs1,
            LibraryPath = config.TransitionRollup.LibraryPath,
            LibraryMinFragments = config.TransitionRollup.LibraryMinFragments,
            LibraryMzTolerance = config.TransitionRollup.LibraryMzTolerance,
            LibraryOutlierThreshold = config.TransitionRollup.LibraryOutlierThreshold,
            LibraryRemoveOutliers = config.TransitionRollup.LibraryRemoveOutliers,
            // Residual files are named for the value file they explain and sit beside, matching
            // the Python engine: peptides_rollup.parquet -> peptides_rollup_residuals.parquet.
            // (Through dotnet-v26.13.0 this was "peptide_residuals.parquet", which collided with
            // the peptide-row residuals the protein rollup now writes.) Rows here are TRANSITIONS.
            ResidualsPath = config.Output.IncludeResiduals
                ? Path.Combine(outputDir, "peptides_rollup_residuals.parquet")
                : null,
            MaxDegreeOfParallelism = config.Processing.NWorkers,
            FlushRows = config.Processing.PeptideBatchSize,
            MemoryBudgetMb = config.Processing.MergeMemoryMb,
        };
        if (transitionCfg.Method == TransitionRollupMethod.LibraryAssist)
            report($"  Library-assisted rollup using spectral library: {config.TransitionRollup.LibraryPath}");
        // The merge's own fingerprint anchors the chain: everything downstream is invalid when the
        // inputs change, without each stage having to re-examine them.
        var mergeFp = StageCache.Fingerprint(
            StageDependencies.Merge, config, extraInputs: new[] { fingerprint });
        var rollupFp = StageCache.Fingerprint(
            StageDependencies.TransitionRollup, config, upstream: new[] { mergeFp });

        if (stageCache.CanReuse(StageDependencies.TransitionRollup, rollupFp))
        {
            report($"  Reusing peptides_rollup.parquet ({RowsOf(peptidesRollupPath)}; inputs and "
                + "transition_rollup settings unchanged).");
        }
        else
        {
            var dop = config.Processing.NWorkers <= 0
                ? Environment.ProcessorCount
                : Math.Min(config.Processing.NWorkers, Environment.ProcessorCount);
            report($"  Rollup workers: {dop} thread(s) (streamed to parquet in row-group batches of "
                + $"{Math.Max(1, config.Processing.PeptideBatchSize):N0}).");
            stageCache.Invalidate(StageDependencies.TransitionRollup);
            var t2 = TransitionRollup.Run(
                dataset, cols, transitionCfg, peptidesRollupPath, samples, cancellationToken);
            report($"  Rolled up to {t2.NPeptides:N0} peptides ({t2.NFiltered:N0} filtered below min_transitions).");
            if (transitionCfg.ResidualsPath is not null && transitionCfg.Method == TransitionRollupMethod.MedianPolish)
                report("  Wrote peptides_rollup_residuals.parquet (per-transition median-polish residuals).");
            stageCache.Record(
                StageDependencies.TransitionRollup, rollupFp,
                peptidesRollupPath, transitionCfg.ResidualsPath);
        }

        // Stage 2a: peptide-matrix density diagnostic + optional sample outlier detection.
        {
            // Column-at-a-time, not ParquetTable.Load: the whole-table load materializes every sample
            // column as a nullable double?[] (16 bytes per cell) and then this matrix copies it into
            // 8 more, so both are live at ~24 bytes per cell - 17 GB on a 100-document cohort, for a
            // diagnostic. Reading one column at a time costs the matrix plus a single column.
            using var pepReader = ParquetColumnReader.Open(peptidesRollupPath);
            var m = new double[pepReader.RowCount, samples.Count];
            long nanCells = 0;
            for (var j = 0; j < samples.Count; j++)
            {
                var col = pepReader.ReadDoubles(samples[j]);
                for (var i = 0; i < col.Length; i++)
                {
                    m[i, j] = col[i];
                    if (double.IsNaN(m[i, j]))
                        nanCells++;
                }
            }
            // Should be ~0: Skyline integrates every transition at (imputed) peak boundaries, so
            // its export is already complete; PRISM only floors the rare 0 / #N/A to a small value
            // before rollup. A non-trivial count therefore flags a real data issue (a report
            // missing transitions, a bad column mapping, etc.), not normal missingness.
            var totalCells = (long)pepReader.RowCount * samples.Count;
            report($"  Peptide matrix: {nanCells:N0} missing of {totalCells:N0} cells "
                + $"({(totalCells > 0 ? 100.0 * nanCells / totalCells : 0):0.###}%) "
                + (nanCells == 0 ? "- fully dense, as expected." : "- unexpected; investigate."));

            if (config.SampleOutlierDetection.Enabled)
            {
            var odMethod = config.SampleOutlierDetection.Method == "fold_median"
                ? OutlierDetector.Method.FoldMedian : OutlierDetector.Method.Iqr;
            var od = OutlierDetector.Detect(m, samples, odMethod,
                config.SampleOutlierDetection.IqrMultiplier, config.SampleOutlierDetection.FoldThreshold);

            if (od.Outliers.Count == 0)
            {
                report("  Sample outlier detection: no low-signal outliers.");
            }
            else if (config.SampleOutlierDetection.Action == "exclude")
            {
                report($"  Sample outlier detection: EXCLUDING {od.Outliers.Count} low-signal sample(s): "
                    + string.Join(", ", od.Outliers));
                var excluded = new HashSet<string>(od.Outliers, StringComparer.Ordinal);
                var keptSamples = new List<string>(samples.Count);
                var keptBatch = new List<string>(samples.Count);
                for (var j = 0; j < samples.Count; j++)
                {
                    if (excluded.Contains(samples[j]))
                        continue;
                    keptSamples.Add(samples[j]);
                    keptBatch.Add(batchLabels[j]);
                }
                samples = keptSamples;
                batchLabels = keptBatch;
                batches = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
                var mb = batches.Count >= 2;
                peptideCombat = config.BatchCorrection.Enabled && config.BatchCorrection.PeptideLevel && mb;
                proteinCombat = config.BatchCorrection.Enabled && config.BatchCorrection.ProteinLevel && mb;
            }
            else
            {
                report($"  Sample outlier detection: {od.Outliers.Count} low-signal sample(s) flagged "
                    + "(report only, kept): " + string.Join(", ", od.Outliers));
            }
            }
        }

        // Stage 2b/2c: peptide normalization + ComBat -> peptides_log2_internal (LOG2) +
        // corrected_peptides (LINEAR).
        // Reference-sample mask (for reference-anchored ComBat), aligned to the current samples.
        var refType = config.BatchCorrection.ReferenceType ?? "reference";
        var referenceAnchored = config.BatchCorrection.ReferenceAnchored;
        var refMask = samples
            .Select(s => string.Equals(resolvedType[s], refType, StringComparison.OrdinalIgnoreCase))
            .ToList();

        // Control-sample column indices for the before/after median-CV report (computed on linear scale).
        var refIdx = Enumerable.Range(0, samples.Count)
            .Where(j => string.Equals(resolvedType[samples[j]], "reference", StringComparison.OrdinalIgnoreCase))
            .ToList();
        var qcIdx = Enumerable.Range(0, samples.Count)
            .Where(j => string.Equals(resolvedType[samples[j]], "qc", StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (referenceAnchored && (peptideCombat || proteinCombat))
        {
            var nRefSamples = refMask.Count(x => x);
            report(nRefSamples > 0
                ? $"  ComBat: reference-anchored using {nRefSamples} '{refType}' sample(s)."
                : $"  ComBat: reference-anchored requested but no '{refType}' samples found; using standard ComBat.");
        }

        // Parsimony is computed HERE, before the peptide output is written, because that output carries
        // the protein groups each peptide belongs to (so the peptide and protein tables can be joined,
        // and a peptide can be navigated to in Skyline). Its banner and CSV stay at Stage 3 below, where
        // the grouping is reported; only the computation moved.
        var groupsPath = Path.Combine(outputDir, "protein_groups.csv");
        // Recomputed on every run, deliberately - see the "not cached" note in StageDependencies.
        // protein_groups.csv keeps only the COUNT of each group's all-mapped peptides, so a group read
        // back from it has AllMappedPeptides = the parsimony-ASSIGNED set. That is the list the default
        // shared_peptide_handling quantifies from, so reusing it would drop every shared peptide from
        // every protein and quietly change the answer. It is also the cheapest of the heavy stages.
        var groups = ParsimonyEngine.Run(
            dataset, cols, config.Parsimony.Enabled, config.Parsimony.FastaPath,
            config.Parsimony.Enzyme, config.Parsimony.EnzymeSpecificity,
            config.Processing.MergeMemoryMb);
        var pepNormFp = StageCache.Fingerprint(
            StageDependencies.PeptideNormalize, config,
            upstream: new[] { rollupFp },
            // The resolved sample types and batches are an INPUT to this stage and are not config: they
            // come from the metadata files and the estimator. Fold them in directly, so re-running with
            // a corrected Sample Type in Skyline invalidates the correction that used the old one.
            extraInputs: new[] { SampleContextKey(samples, batchLabels, resolvedType) });

        // Only when the peptide output will actually be written: the index is one of the two
        // memory-heavy structures in this method (a list per peptide, across every group), and a
        // re-run that reuses the peptide arm has no use for it.
        var willWritePeptides = !stageCache.CanReuse(StageDependencies.PeptideNormalize, pepNormFp);
        var peptideGroups = willWritePeptides
            ? PeptideGroupIndex(groups)
            : new Dictionary<string, List<ProteinGroup>>(StringComparer.Ordinal);
        // Counted now so the index itself can be dropped as soon as the peptide output is written,
        // rather than staying alive through the protein rollup for the sake of one number.
        var sharedPeptides = peptideGroups.Count(kv => kv.Value.Count > 1);

        cancellationToken.ThrowIfCancellationRequested();
        report($"Stage 2b: Peptide normalization ({config.GlobalNormalization.Method})"
            + (peptideCombat ? " + 2c: ComBat batch correction" : "") + "...");
        var internalPath = Path.Combine(outputDir, "peptides_log2_internal.parquet");
        var correctedPepPath = Path.Combine(outputDir, "corrected_peptides." + config.Output.Format);
        int nPeptides;
        var pepResidualPaths = config.Output.IncludeResiduals
            ? new[]
            {
                Path.Combine(outputDir, "peptides_rollup_residuals.parquet"),
                Path.Combine(outputDir, "corrected_peptides_residuals.parquet"),
            }
            : Array.Empty<string>();
        if (stageCache.CanReuse(StageDependencies.PeptideNormalize, pepNormFp))
        {
            // Always parquet, whatever output.format says - it is PRISM's own intermediate.
            nPeptides = ParquetColumnReader.RowCountOf(internalPath);
            report($"  Reusing peptides_log2_internal / corrected_peptides ({nPeptides:N0} peptides; "
                + "rollup, normalization and batch-correction settings unchanged).");
        }
        else
        {
        stageCache.Invalidate(StageDependencies.PeptideNormalize);
        nPeptides = NormalizeCorrectStage.Run(new NormalizeCorrectRequest
        {
            WideParquet = peptidesRollupPath,
            MetaSpec = new[]
            {
                (cols.Peptide, MetaType.Str), ("n_transitions", MetaType.Long), ("mean_rt", MetaType.Double),
            },
            Samples = samples,
            BatchLabels = batchLabels,
            // Corrected residuals for the peptide arm: rows are transitions, features are peptides.
            RawResidualsPath = config.Output.IncludeResiduals
                ? Path.Combine(outputDir, "peptides_rollup_residuals.parquet")
                : null,
            CorrectedResidualsPath = config.Output.IncludeResiduals
                ? Path.Combine(outputDir, "corrected_peptides_residuals.parquet")
                : null,
            CombatEnabled = peptideCombat,
            NormMethod = config.GlobalNormalization.Method,
            InternalLog2Path = internalPath,
            CorrectedLinearPath = correctedPepPath,
            Report = report,
            RefIdx = refIdx,
            QcIdx = qcIdx,
            ReferenceAnchored = referenceAnchored,
            ReferenceMask = refMask,
            RtColumn = "mean_rt",
            RtLowessFrac = config.GlobalNormalization.RtLowess.Frac,
            RtLowessGridPoints = config.GlobalNormalization.RtLowess.NGridPoints,
            AutoRevert = config.BatchCorrection.AutoRevert,
            // Stamped onto the CORRECTED peptide output only (see NormalizeCorrectStage): the internal
            // log2 file feeds the protein rollup and the QC report, whose readers treat any undeclared
            // column as a sample.
            DerivedMeta = PeptideGroupColumns(peptideGroups),
            DerivedKeyColumn = cols.Peptide,
            PathReport = path => report($"  Path: {path}."),
            CancellationToken = cancellationToken,
        });
        report($"  Wrote {nPeptides:N0} corrected peptides.");
        stageCache.Record(
            StageDependencies.PeptideNormalize, pepNormFp,
            new[] { internalPath, correctedPepPath }.Concat(pepResidualPaths).ToArray());
        }
        // Done with the index; the protein rollup that follows is the other memory-heavy stage, so let
        // this go rather than holding it alongside another full matrix.
        peptideGroups = null!;

        // Stage 3: parsimony.
        report("============================================================");
        report(config.Parsimony.Enabled ? "Stage 3: Protein parsimony" : "Stage 3: Protein grouping (parsimony disabled)");
        report("============================================================");
        if (!string.IsNullOrWhiteSpace(config.Parsimony.FastaPath))
            report($"  FASTA-based parsimony map: {config.Parsimony.FastaPath} "
                + $"(enzyme={config.Parsimony.Enzyme}, specificity={config.Parsimony.EnzymeSpecificity})");
        // Written on both paths (it is a small CSV), so the file the cache vouches for is always
        // present - including the first run after an older release, whose sidecar this one ignores.
        ProteinGroupsCsv.Write(groups, groupsPath);
        report($"  {(config.Parsimony.Enabled ? "Computed" : "Built")} {groups.Count:N0} protein groups.");
        if (sharedPeptides > 0)
            report($"  {sharedPeptides:N0} peptide(s) map to more than one group; corrected_peptides lists "
                + $"all of them, '{PeptideGroupSeparator}'-separated.");

        // Stage 4: peptide -> protein.
        report("============================================================");
        cancellationToken.ThrowIfCancellationRequested();
        report($"Stage 4: Peptide -> protein rollup ({config.ProteinRollup.Method})");
        report("============================================================");
        var proteinsRawPath = Path.Combine(outputDir, "proteins_raw.parquet");
        var proteinCfg = new ProteinRollupConfig
        {
            Method = config.ProteinRollup.Method switch
            {
                "sum" => ProteinRollupMethod.Sum,
                "topn" => ProteinRollupMethod.TopN,
                "maxlfq" => ProteinRollupMethod.MaxLfq,
                "ibaq" => ProteinRollupMethod.Ibaq,
                _ => ProteinRollupMethod.MedianPolish,
            },
            MinPeptides = config.ProteinRollup.MinPeptides,
            TopN = config.ProteinRollup.Topn.N,
            TopNSelection = config.ProteinRollup.Topn.Selection,
            SharedPeptideHandling = config.Parsimony.SharedPeptideHandling,
            // proteins_raw_residuals.parquet: rows are PEPTIDES - each peptide's deviation from
            // its protein group's fitted profile, beside the proteins_raw.parquet it explains.
            // Stage 4c derives corrected_proteins_residuals.parquet from it.
            ResidualsPath = config.Output.IncludeResiduals
                ? Path.Combine(outputDir, "proteins_raw_residuals.parquet")
                : null,
        };

        // iBAQ needs a theoretical peptide count per leading protein (from an in-silico FASTA digest).
        IReadOnlyDictionary<string, int>? theoreticalCounts = null;
        if (proteinCfg.Method == ProteinRollupMethod.Ibaq)
        {
            var ibaqFasta = config.ProteinRollup.Ibaq.FastaPath ?? config.Parsimony.FastaPath;
            if (string.IsNullOrWhiteSpace(ibaqFasta))
            {
                report("  iBAQ: no FASTA (protein_rollup.ibaq.fasta_path / parsimony.fasta_path) - "
                    + "falling back to observed peptide counts.");
            }
            else
            {
                var leading = new HashSet<string>(groups.Select(g => g.LeadingProtein), StringComparer.Ordinal);
                theoreticalCounts = FastaParser.GetTheoreticalPeptideCounts(
                    ibaqFasta, leading, config.ProteinRollup.Ibaq.Enzyme, config.ProteinRollup.Ibaq.MissedCleavages,
                    config.ProteinRollup.Ibaq.MinPeptideLength, config.ProteinRollup.Ibaq.MaxPeptideLength);
                report($"  iBAQ: theoretical peptide counts for {theoreticalCounts.Count:N0} proteins "
                    + $"(enzyme={config.ProteinRollup.Ibaq.Enzyme}, missed_cleavages={config.ProteinRollup.Ibaq.MissedCleavages}, "
                    + $"length {config.ProteinRollup.Ibaq.MinPeptideLength}-{config.ProteinRollup.Ibaq.MaxPeptideLength}).");
            }
        }
        var protRollupFp = StageCache.Fingerprint(
            StageDependencies.ProteinRollup, config,
            upstream: new[] { pepNormFp },
            extraInputs: new[] { SampleContextKey(samples, batchLabels, resolvedType) });
        if (stageCache.CanReuse(StageDependencies.ProteinRollup, protRollupFp))
        {
            report($"  Reusing proteins_raw.parquet ({RowsOf(proteinsRawPath)}; peptide matrix, "
                + "grouping and protein_rollup settings unchanged).");
        }
        else
        {
            stageCache.Invalidate(StageDependencies.ProteinRollup);
            var protResult = ProteinRollup.Run(
                internalPath, groups, proteinCfg, cols.Peptide, proteinsRawPath, samples, theoreticalCounts,
                maxDegreeOfParallelism: config.Processing.NWorkers);
            report($"  Rolled up to {protResult.NProteins:N0} proteins.");
            stageCache.Record(
                StageDependencies.ProteinRollup, protRollupFp, proteinsRawPath, proteinCfg.ResidualsPath);
        }
        report($"Stage 4b: Protein normalization ({config.ProteinNormalization.Method})"
            + (proteinCombat ? " + 4c: ComBat" : "") + "...");

        // Stage 4b/4c: protein normalization + ComBat -> corrected_proteins (LINEAR).
        var correctedProtPath = Path.Combine(outputDir, "corrected_proteins." + config.Output.Format);
        var proteinMeta = new (string, MetaType)[]
        {
            ("protein_group", MetaType.Str), ("leading_protein", MetaType.Str), ("leading_name", MetaType.Str),
            ("leading_uniprot_id", MetaType.Str), ("leading_gene_name", MetaType.Str),
            ("leading_description", MetaType.Str), ("n_peptides", MetaType.Long),
            ("n_unique_peptides", MetaType.Long), ("low_confidence", MetaType.Bool),
        };
        var protNormFp = StageCache.Fingerprint(
            StageDependencies.ProteinNormalize, config,
            upstream: new[] { protRollupFp },
            extraInputs: new[] { SampleContextKey(samples, batchLabels, resolvedType) });
        var protResidualPaths = config.Output.IncludeResiduals
            ? new[]
            {
                Path.Combine(outputDir, "proteins_raw_residuals.parquet"),
                Path.Combine(outputDir, "corrected_proteins_residuals.parquet"),
            }
            : Array.Empty<string>();
        int nProteins;
        if (stageCache.CanReuse(StageDependencies.ProteinNormalize, protNormFp))
        {
            // proteins_raw is PRISM's own parquet; corrected_proteins honours output.format and may be
            // CSV, which the parquet reader cannot open - count from the one that is always parquet.
            nProteins = ParquetColumnReader.RowCountOf(proteinsRawPath);
            report($"  Reusing corrected_proteins ({nProteins:N0} proteins; protein matrix, "
                + "normalization and batch-correction settings unchanged).");
        }
        else
        {
        stageCache.Invalidate(StageDependencies.ProteinNormalize);
        nProteins = NormalizeCorrectStage.Run(new NormalizeCorrectRequest
        {
            WideParquet = proteinsRawPath,
            MetaSpec = proteinMeta,
            Samples = samples,
            BatchLabels = batchLabels,
            // Corrected residuals for the protein arm: rows are peptides, features are groups.
            RawResidualsPath = config.Output.IncludeResiduals
                ? Path.Combine(outputDir, "proteins_raw_residuals.parquet")
                : null,
            CorrectedResidualsPath = config.Output.IncludeResiduals
                ? Path.Combine(outputDir, "corrected_proteins_residuals.parquet")
                : null,
            CombatEnabled = proteinCombat,
            NormMethod = config.ProteinNormalization.Method,
            InternalLog2Path = null,
            CorrectedLinearPath = correctedProtPath,
            Report = report,
            RefIdx = refIdx,
            QcIdx = qcIdx,
            ReferenceAnchored = referenceAnchored,
            ReferenceMask = refMask,
            AutoRevert = config.BatchCorrection.AutoRevert,
            PathReport = path => report($"  Path: {path}."),
            CancellationToken = cancellationToken,
        });
        stageCache.Record(
            StageDependencies.ProteinNormalize, protNormFp,
            new[] { correctedProtPath }.Concat(protResidualPaths).ToArray());
        }

        // Stage 5a: marker normalization, on both corrected outputs. Last, because the score has to
        // come from data whose loading is already normalized, and from the PROTEIN level - how much
        // marked material a sample contributed is a property of the sample, not of the table.
        if (config.MarkerNormalization.Enabled)
        {
            report("============================================================");
            cancellationToken.ThrowIfCancellationRequested();
            report("Stage 5a: Marker normalization");
            report("============================================================");

            var markerFp = StageCache.Fingerprint(
                StageDependencies.MarkerNormalize, config,
                upstream: new[] { protNormFp, pepNormFp });
            var scorePath = Path.Combine(outputDir, "marker_normalization.csv");

            if (stageCache.CanReuse(StageDependencies.MarkerNormalize, markerFp))
            {
                report("  Reusing the marker-normalized outputs (matrices and marker list unchanged).");
            }
            else
            {
                // Both corrected files are rewritten in place, so a partial previous attempt must not
                // be adjusted a second time: invalidate first, and let a failure leave no entry.
                stageCache.Invalidate(StageDependencies.MarkerNormalize);

                var list = ProteinListSet.Resolve(
                    config.MarkerNormalization.ProteinList,
                    config.MarkerNormalization.ProteinListFile)
                    ?? throw new InvalidOperationException(
                        "marker_normalization.enabled is true but no protein list was resolved.");
                var scoreMethod =
                    string.Equals(config.MarkerNormalization.Method, "mean", StringComparison.OrdinalIgnoreCase)
                        ? MarkerScoreMethod.Mean
                        : MarkerScoreMethod.Pc1;

                var marker = MarkerNormalizeStage.Run(
                    correctedProtPath, correctedPepPath, list, scoreMethod, samples, report);
                MarkerNormalizeStage.WriteScoreCsv(scorePath, samples, marker.Score);
                report($"  Wrote marker_normalization.csv (per-sample score and marker loadings).");
                report("  NOTE: the marker features are kept in both outputs and flagged "
                    + $"'{MarkerNormalizeStage.MarkerColumn}'. Their values are near-flat by "
                    + "construction - exclude them from any result read off these files.");

                stageCache.Record(
                    StageDependencies.MarkerNormalize, markerFp,
                    correctedProtPath, correctedPepPath, scorePath);
            }
        }

        report("============================================================");
        cancellationToken.ThrowIfCancellationRequested();
        report("Stage 5: Output generation");
        report("============================================================");
        WriteSampleMetadata(
            Path.Combine(outputDir, "sample_metadata.csv"), samples, resolvedBatch, resolvedType, metadata);
        // Keep a copy of the search database beside the results. An absolute path describes the run;
        // it does not let anyone repeat it once the database has been reorganized or the output
        // directory handed to someone else.
        var archivedFasta = FastaArchive.Archive(config, outputDir, report);
        Provenance.Write(
            Path.Combine(outputDir, "parameters.json"), config, inputs,
            new Provenance.Stats(samples.Count, nPeptides, nProteins, groups.Count),
            DateTime.UtcNow.ToString("o"), archivedFasta);
        var nRef = resolvedType.Values.Count(t => t == "reference");
        var nQc = resolvedType.Values.Count(t => t == "qc");
        var nExp = resolvedType.Values.Count(t => t == "experimental");
        var nBlank = resolvedType.Count - nRef - nQc - nExp;
        report($"  Sample types: {nRef} reference, {nQc} qc, {nExp} experimental"
            + (nBlank > 0 ? $", {nBlank} blank/solvent (excluded from groupings)" : "") + ".");
        report($"  Wrote corrected_peptides / corrected_proteins ({config.Output.Format}, linear), "
            + "sample_metadata.csv, and parameters.json (provenance).");

        // Stage 5b: QC report.
        if (config.QcReport.Enabled)
        {
            report("Stage 5b: Generating QC report (qc_report.html)...");
            // The log sink matters here: MS2 signal accounting runs inside the report and is the one
            // part of it that can take minutes, so its progress and its skip reasons belong in the run log.
            QcReport.Generate(outputDir, config, savePlots: config.QcReport.SavePlots, log: report);
        }

        EndStage();
        totalTimer.Stop();
        reportRaw("============================================================");
        reportRaw($"Stage timings (total {Fmt(totalTimer.Elapsed)})");
        reportRaw("============================================================");
        foreach (var (stage, elapsed) in timings.OrderByDescending(t => t.Elapsed))
            reportRaw($"  {stage,-12} {Fmt(elapsed),9}  "
                + $"{(totalTimer.Elapsed.TotalSeconds > 0 ? 100 * elapsed.TotalSeconds / totalTimer.Elapsed.TotalSeconds : 0),5:n1}%");

        report($"PRISM complete: {nPeptides:N0} peptides, {nProteins:N0} proteins, {samples.Count:N0} samples, "
            + $"{batches.Count} batch(es).");
        return new Result(nPeptides, nProteins, samples.Count, batches);
    }

    /// <summary>
    /// Separator between the protein groups of a shared peptide in the corrected peptide output. A
    /// peptide that maps to several groups lists them all rather than being arbitrarily assigned to one -
    /// which group "owns" it is a quantification decision (parsimony.shared_peptide_handling), not a fact
    /// about the peptide.
    /// </summary>
    public const string PeptideGroupSeparator = ";";

    /// <summary>Peptide -> every protein group that peptide maps to, in group order.</summary>
    private static Dictionary<string, List<ProteinGroup>> PeptideGroupIndex(IReadOnlyList<ProteinGroup> groups)
    {
        var index = new Dictionary<string, List<ProteinGroup>>(StringComparer.Ordinal);
        foreach (var group in groups)
        {
            // AllMappedPeptides, not Peptides: the question here is "which groups contain this peptide",
            // which includes the shared ones parsimony did not assign to this group.
            foreach (var peptide in group.AllMappedPeptides)
            {
                if (!index.TryGetValue(peptide, out var list))
                    index[peptide] = list = new List<ProteinGroup>();
                if (!list.Contains(group))
                    list.Add(group);
            }
        }
        return index;
    }

    /// <summary>
    /// The protein-group columns stamped onto corrected_peptides: the group IDs (join key to
    /// corrected_proteins) and the leading protein/name (what identifies the protein in Skyline).
    /// </summary>
    private static IReadOnlyList<(string Name, Func<string, string> Value)> PeptideGroupColumns(
        IReadOnlyDictionary<string, List<ProteinGroup>> peptideGroups)
    {
        string Join(string peptide, Func<ProteinGroup, string> field) =>
            peptideGroups.TryGetValue(peptide, out var list)
                ? string.Join(PeptideGroupSeparator, list.Select(field))
                : "";

        return new (string, Func<string, string>)[]
        {
            ("protein_group", p => Join(p, g => g.GroupId)),
            ("leading_protein", p => Join(p, g => g.LeadingProtein)),
            ("leading_name", p => Join(p, g => g.LeadingName)),
            ("leading_gene_name", p => Join(p, g => g.LeadingGeneName)),
        };
    }

    private static Dictionary<string, string> GetBatchMap(MergedDataset dataset, SkylineColumns cols)
    {
        var batchCol = cols.Batch ?? "Batch";
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn, DuckDbMerge.AutoMemoryBudgetMb(), DuckDbMerge.ResolveTempDirectory(dataset.Root));
        using var cmd = DuckDbTuning.StreamingCommand(conn,
            $"SELECT DISTINCT \"{cols.Sample}\" AS s, \"{batchCol}\" AS b "
            + $"FROM {MergedParquetReader.Scan(dataset.ScanTarget)}");
        using var reader = cmd.ExecuteReader();
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        while (reader.Read())
        {
            if (reader.IsDBNull(0))
                continue;
            map[reader.GetString(0)] = reader.IsDBNull(1)
                ? "batch1"
                : Convert.ToString(reader.GetValue(1), CultureInfo.InvariantCulture) ?? "batch1";
        }
        return map;
    }

    private static string SampleContextKey(
        IReadOnlyList<string> samples, IReadOnlyList<string> batchLabels,
        IReadOnlyDictionary<string, string> resolvedType)
    {
        var sb = new StringBuilder();
        for (var i = 0; i < samples.Count; i++)
            sb.Append(samples[i]).Append('\t')
              .Append(i < batchLabels.Count ? batchLabels[i] : "").Append('\t')
              .Append(resolvedType.GetValueOrDefault(samples[i], "experimental")).Append('\n');
        return Convert.ToHexString(
            System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())));
    }

    /// <summary>
    /// Write sample_metadata.csv: PRISM's four resolved fields, then EVERY other column of the
    /// Replicates report (or of the metadata files given to the CLI), verbatim.
    /// <para>
    /// The annotations are the study - Subject, Timepoint, responder status, days between draws - and
    /// this file is where an analyst looks for them next to the abundances. PRISM interprets three of
    /// the report's columns and used to write only those, so a cohort's design reached the output
    /// directory only inside a Skyline export that most downstream code never opens. The four resolved
    /// columns keep their names and positions, so anything already reading this file still works.
    /// </para>
    /// <para>
    /// The report's own `SampleType` is carried too, even though `sample_type` is derived from it,
    /// because the derivation is LOSSY and the raw value is its own fact. Skyline's sample types are an
    /// editable list: `Solvent`, `Blank` and `Double Blank` are three different things that PRISM maps
    /// onto one `blank`, and a type a user added to the dropdown maps onto `experimental` along with
    /// `Unknown`. `sample_type` says what PRISM did with the replicate; `SampleType` says what the
    /// document calls it, and only the second survives a custom vocabulary. The same holds for a
    /// `Batch` column, whose raw value differs from the resolved `batch` whenever PRISM fell back
    /// (blank cell, `#N/A`, or no batch annotation at all).
    /// </para>
    /// </summary>
    /// <summary>
    /// The resolved per-sample context that the normalize/correct stages consume but the config does
    /// not describe: which samples there are, each one's batch, and each one's sample type. These come
    /// from the metadata files and the batch estimator, so a corrected Sample Type in Skyline - or a
    /// sample dropped as an outlier - has to invalidate a cached correction even though no config key
    /// moved.
    /// </summary>
    /// <summary>
    /// "N peptides" for a log line, from a parquet file's footer - or "reused" when the file is not
    /// parquet (output.format: csv/tsv), where opening it as parquet would throw.
    /// </summary>
    private static string RowsOf(string path)
    {
        try
        {
            return path.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase)
                ? $"{ParquetColumnReader.RowCountOf(path):N0} rows"
                : "unchanged";
        }
        catch
        {
            return "unchanged";
        }
    }

    private static void WriteSampleMetadata(
        string path, IReadOnlyList<string> samples,
        IReadOnlyDictionary<string, string> resolvedBatch, IReadOnlyDictionary<string, string> resolvedType,
        ReplicateMetadata? metadata)
    {
        // Every report column goes out. Only an EXACT name clash with one of the four reserved fields
        // is adjusted, and by suffixing rather than dropping: two columns with the same header would
        // resolve to whichever a reader indexes first, silently, while dropping one would lose data the
        // run was asked to carry. (A replicate annotation really can be called "batch".)
        var reserved = new[] { "sample_id", "sample", "sample_type", "batch" };
        var extraCols = new List<(string Header, string Column)>();
        foreach (var col in metadata?.ColumnNames ?? new List<string>())
        {
            var header = reserved.Contains(col, StringComparer.OrdinalIgnoreCase)
                ? col + " (report)"
                : col;
            extraCols.Add((header, col));
        }

        var sb = new StringBuilder("sample_id,sample,sample_type,batch");
        foreach (var (header, _) in extraCols)
            sb.Append(',').Append(Csv(header));
        sb.Append('\n');

        foreach (var sampleId in samples)
        {
            var replicate = SampleIdToReplicate(sampleId);
            var batch = resolvedBatch.GetValueOrDefault(sampleId, "batch1");
            var type = resolvedType.GetValueOrDefault(sampleId, "experimental");
            sb.Append(Csv(sampleId)).Append(',').Append(Csv(replicate)).Append(',')
              .Append(type).Append(',').Append(Csv(batch));

            var values = metadata?.ValuesFor(sampleId, replicate);
            foreach (var (_, col) in extraCols)
                sb.Append(',').Append(Csv(values?.GetValueOrDefault(col) ?? ""));
            sb.Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static string ClassifySampleType(string sampleId, string replicate, PrismConfig config)
    {
        bool Matches(IEnumerable<string> patterns) =>
            patterns.Any(p => sampleId.Contains(p, StringComparison.Ordinal)
                              || replicate.Contains(p, StringComparison.Ordinal));
        if (Matches(config.SampleAnnotations.ReferencePattern))
            return "reference";
        if (Matches(config.SampleAnnotations.QcPattern))
            return "qc";
        return "experimental";
    }

    private static string SampleIdToReplicate(string sampleId)
    {
        const string sep = "__@__";
        var idx = sampleId.IndexOf(sep, StringComparison.Ordinal);
        return idx >= 0 ? sampleId[..idx] : sampleId;
    }

    private static string Csv(string s) =>
        s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
}
