using System.Linq;
using SkylinePrism.Core.Config;
using Xunit;

namespace SkylinePrism.Tests.Config;

/// <summary>
/// Guards the config-honesty layer: unrecognized keys are reported (not silently dropped), the
/// nested library_assist block folds onto the flat fields, and unported method choices throw
/// instead of silently doing something else.
/// </summary>
public class ConfigValidationTests
{
    [Fact]
    public void FindUnknownKeys_FlagsTypoAndPythonOnlyKeys()
    {
        const string yaml = """
            transition_rollup:
              method: median_polish
              min_transitions: 3
              learn_adaptive_weights: true   # Python-only (adaptive not ported)
              adaptive_rollup:
                beta_mz: 0.0
            global_normalization:
              method: rt_lowess
              vsn_params:                     # Python-only
                optimize_params: false
            protein_rollup:
              method: median_polish
              min_peptides: 3
              min_peptdes: 2                  # typo
            data:                             # section supported; a bad key inside is still caught
              abundance_column: Area
              bogus_column: x                 # typo
            """;

        var unknown = PrismConfig.FindUnknownKeys(yaml);

        Assert.Contains("transition_rollup.learn_adaptive_weights", unknown);
        Assert.Contains("transition_rollup.adaptive_rollup", unknown);
        Assert.Contains("global_normalization.vsn_params", unknown);
        Assert.Contains("protein_rollup.min_peptdes", unknown);
        Assert.Contains("data.bogus_column", unknown);
        Assert.DoesNotContain("data", unknown);              // the section is recognized now
        Assert.DoesNotContain("data.abundance_column", unknown);
    }

    [Fact]
    public void FindUnknownKeys_AcceptsAllImplementedKeys()
    {
        // Every key the C# pipeline actually reads, including the new/nested ones.
        const string yaml = """
            transition_rollup:
              method: library_assist
              min_transitions: 3
              topn_count: 3
              topn_selection: correlation
              topn_weighting: sqrt
              consensus_regularization: 0.1
              use_ms1: false
              library_path: lib.blib
              library_min_fragments: 3
              library_mz_tolerance: 0.02
              library_outlier_threshold: 1.0
              library_remove_outliers: true
              library_assist:
                library_path: lib.blib
                min_matched_fragments: 3
                mz_tolerance: 0.02
                outlier_threshold: 1.0
                remove_outliers: true
                fitting_method: median_polish
            global_normalization:
              method: rt_lowess
              rt_lowess:
                frac: 0.3
                n_grid_points: 100
            batch_correction:
              enabled: true
              peptide_level: true
              protein_level: true
              method: combat
              reference_anchored: false
              reference_type: reference
            protein_rollup:
              method: ibaq
              min_peptides: 3
              topn:
                n: 3
                selection: frequency
              ibaq:
                fasta_path: db.fasta
                enzyme: trypsin
                missed_cleavages: 0
                min_peptide_length: 6
                max_peptide_length: 30
            protein_normalization:
              method: median
            sample_annotations:
              reference_pattern: ["-Pool_"]
              qc_pattern: ["-QC_"]
            parsimony:
              enabled: true
              fasta_path: null
              shared_peptide_handling: all_groups
            output:
              format: parquet
              include_residuals: true
            qc_report:
              enabled: true
              save_plots: true
            sample_outlier_detection:
              enabled: true
              action: report
              method: iqr
              iqr_multiplier: 1.5
              fold_threshold: 0.1
            metadata:
              batch_column: Batch
              sample_type_column: Sample Type
            processing:
              n_workers: 0
              peptide_batch_size: 2000
            batch_estimation:
              method: auto
              n_batches: 3
              gap_iqr_multiplier: 1.5
            """;

        var unknown = PrismConfig.FindUnknownKeys(yaml);

        Assert.Empty(unknown);
    }

    [Fact]
    public void ResolveLibraryAssist_NestedBlockFoldsOntoFlatFields()
    {
        const string yaml = """
            transition_rollup:
              method: library_assist
              library_assist:
                library_path: nested.blib
                min_matched_fragments: 5
                mz_tolerance: 0.05
                outlier_threshold: 2.0
                remove_outliers: false
            """;

        var config = PrismConfig.Parse(yaml); // Parse calls ResolveLibraryAssist

        Assert.Equal("nested.blib", config.TransitionRollup.LibraryPath);
        Assert.Equal(5, config.TransitionRollup.LibraryMinFragments);
        Assert.Equal(0.05, config.TransitionRollup.LibraryMzTolerance);
        Assert.Equal(2.0, config.TransitionRollup.LibraryOutlierThreshold);
        Assert.False(config.TransitionRollup.LibraryRemoveOutliers);
    }

    [Fact]
    public void Validate_ThrowsOnAdaptiveMethod()
    {
        var config = PrismConfig.Parse("transition_rollup:\n  method: adaptive\n");
        var ex = Assert.Throws<System.NotSupportedException>(() => config.Validate());
        Assert.Contains("adaptive", ex.Message);
    }

    [Fact]
    public void Validate_ThrowsOnLeastSquaresFittingMethod()
    {
        var config = PrismConfig.Parse(
            "transition_rollup:\n  method: library_assist\n  library_assist:\n    fitting_method: least_squares\n");
        var ex = Assert.Throws<System.NotSupportedException>(() => config.Validate());
        Assert.Contains("least_squares", ex.Message);
    }

    [Fact]
    public void Validate_ThrowsOnNonCombatBatchMethod()
    {
        var config = PrismConfig.Parse("batch_correction:\n  method: harmony\n");
        Assert.Throws<System.NotSupportedException>(() => config.Validate());
    }

    [Fact]
    public void Validate_PassesOnDefaultConfig()
    {
        new PrismConfig().Validate(); // must not throw
    }

    [Fact]
    public void ConfigTemplate_HasNoUnknownKeysAndValidates()
    {
        // The emitted templates must only use keys the pipeline reads, and must pass validation -
        // guards the template from drifting out of sync with the schema.
        Assert.Empty(PrismConfig.FindUnknownKeys(ConfigTemplate.Default()));
        Assert.Empty(PrismConfig.FindUnknownKeys(ConfigTemplate.Minimal()));
        PrismConfig.Parse(ConfigTemplate.Default()).Validate();
        PrismConfig.Parse(ConfigTemplate.Minimal()).Validate();
    }
}
