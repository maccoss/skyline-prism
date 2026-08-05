using System.Collections.Generic;
using System.Linq;
using YamlDotNet.Serialization;

namespace SkylinePrism.Core.Config;

/// <summary>
/// Writes a <see cref="PrismConfig"/> back out as the SMALLEST YAML that reproduces it.
/// <para>
/// A plain object dump emits every property of every section - roughly 60 keys, most of which do
/// not apply to the selected methods (topn_* while the method is library_assist, the ibaq block
/// while the protein rollup is median_polish, rt_lowess tuning while normalization is median).
/// That is noise, and worse, it reads as though those knobs are in play. This writer emits:
/// </para>
/// <list type="bullet">
///   <item>the keys that define the run (method choices and their operands), always;</item>
///   <item>tuning keys for the SELECTED method only;</item>
///   <item>everything else only when it differs from the built-in default.</item>
/// </list>
/// <para>
/// Keys that NAME an algorithm are always written for an active section, even at their default and
/// even where C# implements only one choice - the config is meant to be handed to the CLI, and the
/// Python engine offers choices C# does not (<c>library_fitting_method</c> is median_polish-only
/// here but median_polish OR least_squares there). Eliding them would leave the reader unable to
/// tell which algorithm produced the numbers. Only numeric and boolean tuning values are elided.
/// </para>
/// <para>
/// Omitted keys fall back to exactly these defaults when the config is re-read, so the emitted
/// YAML is behaviorally identical to the config it came from (asserted by ConfigWriterTests).
/// Values the selected methods ignore are dropped rather than preserved - a stale library_path
/// carried under <c>method: sum</c> does not survive the round trip, because it changes nothing.
/// </para>
/// </summary>
public static class ConfigWriter
{
    private const string Header =
        "# PRISM configuration - only the settings that apply to the selected methods.\n" +
        "# Anything not listed uses its built-in default; run 'prism config-template' for the\n" +
        "# fully annotated template with every available option.\n";

    /// <summary>Minimal YAML for <paramref name="config"/> (see the type remarks).</summary>
    public static string ToYaml(PrismConfig config)
    {
        var def = new PrismConfig();
        var root = new Dictionary<string, object?>();

        AddSection(root, "data", Data(config.Data, config.Metadata));
        root["transition_rollup"] = TransitionRollup(config.TransitionRollup);
        root["global_normalization"] = GlobalNormalization(config.GlobalNormalization, def.GlobalNormalization);
        root["sample_outlier_detection"] = OutlierDetection(config.SampleOutlierDetection, def.SampleOutlierDetection);
        root["batch_correction"] = BatchCorrection(config.BatchCorrection, def.BatchCorrection);
        root["parsimony"] = Parsimony(config.Parsimony, def.Parsimony);
        root["protein_rollup"] = ProteinRollup(config.ProteinRollup, def.ProteinRollup);
        root["protein_normalization"] = new Dictionary<string, object?>
        {
            ["method"] = config.ProteinNormalization.Method,
        };
        root["qc_report"] = new Dictionary<string, object?>
        {
            ["enabled"] = config.QcReport.Enabled,
            ["save_plots"] = config.QcReport.SavePlots,
        };
        AddSection(root, "output", Output(config.Output, def.Output));
        AddSection(root, "processing", Processing(config.Processing, def.Processing));
        AddSection(root, "batch_estimation", BatchEstimation(config.BatchEstimation, def.BatchEstimation));
        AddSection(root, "sample_annotations", SampleAnnotations(config.SampleAnnotations, def.SampleAnnotations));

        return Header + new SerializerBuilder().Build().Serialize(root);
    }

    private static Dictionary<string, object?> Data(
        PrismConfig.DataSection d, PrismConfig.MetadataSection m)
    {
        var s = new Dictionary<string, object?>();
        AddIfSet(s, "abundance_column", d.AbundanceColumn);
        AddIfSet(s, "rt_column", d.RtColumn);
        AddIfSet(s, "peptide_column", d.PeptideColumn);
        AddIfSet(s, "protein_column", d.ProteinColumn);
        AddIfSet(s, "protein_name_column", d.ProteinNameColumn);
        AddIfSet(s, "sample_column", d.SampleColumn);
        AddIfSet(s, "transition_column", d.TransitionColumn);
        AddIfSet(s, "precursor_column", d.PrecursorColumn);
        AddIfSet(s, "fragment_column", d.FragmentColumn);
        // The batch / sample-type columns are written under data:, which BOTH engines read - C#
        // prefers data.* and falls back to metadata.* (PrismPipeline: Data.BatchColumn ??
        // Metadata.BatchColumn), while the Python engine knows only data.*. Emitting the C#-only
        // metadata: section instead would make every tool-authored config warn on the Python CLI.
        AddIfSet(s, "batch_column", d.BatchColumn ?? m.BatchColumn);
        AddIfSet(s, "sample_type_column", d.SampleTypeColumn ?? m.SampleTypeColumn);
        return s;
    }

    private static Dictionary<string, object?> TransitionRollup(PrismConfig.TransitionRollupSection tr)
    {
        var s = new Dictionary<string, object?>
        {
            ["method"] = tr.Method,
            ["min_transitions"] = tr.MinTransitions,
            ["use_ms1"] = tr.UseMs1,
        };

        switch (tr.Method?.ToLowerInvariant())
        {
            case "topn":
                s["topn_count"] = tr.TopnCount;
                s["topn_selection"] = tr.TopnSelection;
                s["topn_weighting"] = tr.TopnWeighting;
                break;
            case "consensus":
                s["consensus_regularization"] = tr.ConsensusRegularization;
                break;
            case "library_assist":
                AddIfSet(s, "library_path", tr.LibraryPath);
                s["library_min_fragments"] = tr.LibraryMinFragments;
                s["library_mz_tolerance"] = tr.LibraryMzTolerance;
                s["library_outlier_threshold"] = tr.LibraryOutlierThreshold;
                s["library_remove_outliers"] = tr.LibraryRemoveOutliers;
                // Always written: C# implements only median_polish (least_squares aborts), but Python
                // implements both, so the config has to say which fit produced the sample scale.
                s["library_fitting_method"] = tr.LibraryFittingMethod;
                break;
        }
        return s;
    }

    private static Dictionary<string, object?> GlobalNormalization(
        PrismConfig.GlobalNormalizationSection gn, PrismConfig.GlobalNormalizationSection def)
    {
        var s = new Dictionary<string, object?> { ["method"] = gn.Method };
        if (gn.Method?.ToLowerInvariant() == "rt_lowess")
        {
            var lowess = new Dictionary<string, object?>();
            AddIfChanged(lowess, "frac", gn.RtLowess.Frac, def.RtLowess.Frac);
            AddIfChanged(lowess, "n_grid_points", gn.RtLowess.NGridPoints, def.RtLowess.NGridPoints);
            AddSection(s, "rt_lowess", lowess);
        }
        return s;
    }

    private static Dictionary<string, object?> OutlierDetection(
        PrismConfig.SampleOutlierDetectionSection od, PrismConfig.SampleOutlierDetectionSection def)
    {
        var s = new Dictionary<string, object?> { ["enabled"] = od.Enabled };
        if (!od.Enabled)
            return s;

        s["action"] = od.Action;
        s["method"] = od.Method;
        if (od.Method?.ToLowerInvariant() == "iqr")
            AddIfChanged(s, "iqr_multiplier", od.IqrMultiplier, def.IqrMultiplier);
        else
            AddIfChanged(s, "fold_threshold", od.FoldThreshold, def.FoldThreshold);
        return s;
    }

    private static Dictionary<string, object?> BatchCorrection(
        PrismConfig.BatchCorrectionSection bc, PrismConfig.BatchCorrectionSection def)
    {
        var s = new Dictionary<string, object?> { ["enabled"] = bc.Enabled };
        if (!bc.Enabled)
            return s;

        s["peptide_level"] = bc.PeptideLevel;
        s["protein_level"] = bc.ProteinLevel;
        s["method"] = bc.Method;
        if (bc.ReferenceAnchored)
        {
            s["reference_anchored"] = true;
            s["reference_type"] = bc.ReferenceType;
        }
        AddIfChanged(s, "auto_revert", bc.AutoRevert, def.AutoRevert);
        return s;
    }

    private static Dictionary<string, object?> Parsimony(
        PrismConfig.ParsimonySection p, PrismConfig.ParsimonySection def)
    {
        var s = new Dictionary<string, object?> { ["enabled"] = p.Enabled };
        if (!p.Enabled)
            return s;

        s["shared_peptide_handling"] = p.SharedPeptideHandling;
        // enzyme / enzyme_specificity only apply to the FASTA membership check; without a FASTA the
        // peptide-protein map comes from Skyline's (already enzyme-aware) Protein Accession column.
        if (!string.IsNullOrWhiteSpace(p.FastaPath))
        {
            s["fasta_path"] = p.FastaPath;
            s["enzyme"] = p.Enzyme;
            s["enzyme_specificity"] = p.EnzymeSpecificity;
        }
        return s;
    }

    private static Dictionary<string, object?> ProteinRollup(
        PrismConfig.ProteinRollupSection pr, PrismConfig.ProteinRollupSection def)
    {
        var s = new Dictionary<string, object?>
        {
            ["method"] = pr.Method,
            ["min_peptides"] = pr.MinPeptides,
        };

        switch (pr.Method?.ToLowerInvariant())
        {
            case "topn":
                s["topn"] = new Dictionary<string, object?>
                {
                    ["n"] = pr.Topn.N,
                    ["selection"] = pr.Topn.Selection,
                };
                break;
            case "ibaq":
                var ibaq = new Dictionary<string, object?>();
                AddIfSet(ibaq, "fasta_path", pr.Ibaq.FastaPath);
                ibaq["enzyme"] = pr.Ibaq.Enzyme;
                ibaq["missed_cleavages"] = pr.Ibaq.MissedCleavages;
                AddIfChanged(ibaq, "min_peptide_length", pr.Ibaq.MinPeptideLength, def.Ibaq.MinPeptideLength);
                AddIfChanged(ibaq, "max_peptide_length", pr.Ibaq.MaxPeptideLength, def.Ibaq.MaxPeptideLength);
                s["ibaq"] = ibaq;
                break;
        }
        return s;
    }

    private static Dictionary<string, object?> Output(
        PrismConfig.OutputSection o, PrismConfig.OutputSection def)
    {
        var s = new Dictionary<string, object?>();
        AddIfChanged(s, "format", o.Format, def.Format);
        AddIfChanged(s, "include_residuals", o.IncludeResiduals, def.IncludeResiduals);
        return s;
    }

    private static Dictionary<string, object?> Processing(
        PrismConfig.ProcessingSection p, PrismConfig.ProcessingSection def)
    {
        var s = new Dictionary<string, object?>();
        AddIfChanged(s, "n_workers", p.NWorkers, def.NWorkers);
        AddIfChanged(s, "peptide_batch_size", p.PeptideBatchSize, def.PeptideBatchSize);
        return s;
    }

    private static Dictionary<string, object?> BatchEstimation(
        PrismConfig.BatchEstimationSection be, PrismConfig.BatchEstimationSection def)
    {
        var s = new Dictionary<string, object?>();
        AddIfChanged(s, "method", be.Method, def.Method);
        if (be.NBatches.HasValue)
            s["n_batches"] = be.NBatches.Value;
        AddIfChanged(s, "gap_iqr_multiplier", be.GapIqrMultiplier, def.GapIqrMultiplier);
        return s;
    }

    private static Dictionary<string, object?> SampleAnnotations(
        PrismConfig.SampleAnnotationsSection sa, PrismConfig.SampleAnnotationsSection def)
    {
        var s = new Dictionary<string, object?>();
        if (!sa.ReferencePattern.SequenceEqual(def.ReferencePattern))
            s["reference_pattern"] = sa.ReferencePattern;
        if (!sa.QcPattern.SequenceEqual(def.QcPattern))
            s["qc_pattern"] = sa.QcPattern;
        return s;
    }

    /// <summary>Add a subsection only when it has content, so empty mappings are never emitted.</summary>
    private static void AddSection(
        Dictionary<string, object?> parent, string key, Dictionary<string, object?> section)
    {
        if (section.Count > 0)
            parent[key] = section;
    }

    private static void AddIfSet(Dictionary<string, object?> s, string key, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
            s[key] = value;
    }

    private static void AddIfChanged<T>(Dictionary<string, object?> s, string key, T value, T defaultValue)
    {
        if (!EqualityComparer<T>.Default.Equals(value, defaultValue))
            s[key] = value;
    }
}
