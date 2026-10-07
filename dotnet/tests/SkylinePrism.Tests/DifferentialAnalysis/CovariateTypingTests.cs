using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// <see cref="CovariateTyping"/> and what the design does with each type. The case that motivated
/// it: a longitudinal cohort whose patient IDs are the integers 1-15. Every column of numbers used
/// to be fitted as one centered column, so "adjust for Patient" put a straight-line effect of an ID
/// number into every model and said nothing.
/// </summary>
public class CovariateTypingTests
{
    private static string?[] Repeat(params (string Value, int Times)[] runs) =>
        runs.SelectMany(r => Enumerable.Repeat((string?)r.Value, r.Times)).ToArray();

    [Theory]
    // A name word, with whole numbers that repeat (a subject's ID over its visits): labels.
    [InlineData("Patient (paper Table S3)", "1,1,1,2,2,3,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15", CovariateKind.Categorical)]
    [InlineData("PatientID", "101,101,102,102,103,103,104,104,105,105,106,106,107,107,108,108,109,109,110,110,111,111", CovariateKind.Categorical)]
    [InlineData("File set", "1,2,3", CovariateKind.Categorical)]
    [InlineData("Digestion cycle", "1,2,3", CovariateKind.Categorical)]
    [InlineData("Plate2", "1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12", CovariateKind.Categorical)]
    // A name word on a measurement: decimals, or whole numbers that never repeat, stay numeric.
    [InlineData("Patient age", "65.3,40.7,55.1,61.2,48.9,70.4,52.0,58.6,44.3,67.1,59.9", CovariateKind.Numeric)]
    [InlineData("Run order", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20", CovariateKind.Numeric)]
    [InlineData("Cycle time", "1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5", CovariateKind.Numeric)]
    // Whole words only: these contain a listed word inside another word.
    [InlineData("Onset site", "1.5,2.25,3.75,4.5,5.5,6.5,7.25,8.5,9.5,10.5,11.5", CovariateKind.Numeric)]
    [InlineData("Hybridization", "0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5", CovariateKind.Numeric)]
    // Few whole-number levels: categories.
    [InlineData("Visit", "1,2,3,4,5,6,7", CovariateKind.Categorical)]
    [InlineData("Treated", "0,1,0,1", CovariateKind.Categorical)]
    [InlineData("Dose", "1.0,2.0,1.0,3.0", CovariateKind.Categorical)]
    // Quantities stay numeric.
    [InlineData("ALSFRS-R Total", "48,45,44,40,39,36,33,30,27,24,21", CovariateKind.Numeric)]
    [InlineData("Age at first visit", "65.3,40.7,55.1", CovariateKind.Numeric)]
    // Text is categories, whatever the name.
    [InlineData("Sex", "F,M,F", CovariateKind.Categorical)]
    public void Infer(string name, string csv, CovariateKind expected)
    {
        var guess = CovariateTyping.Infer(name, csv.Split(','));
        Assert.Equal(expected, guess.Kind);
        Assert.False(string.IsNullOrEmpty(guess.Reason));
    }

    [Fact]
    public void TenWholeNumberLevelsAreCategorical_ElevenAreNumeric()
    {
        var ten = Enumerable.Range(1, CovariateTyping.MaxCategoricalLevels).Select(i => (string?)i.ToString()).ToArray();
        var eleven = Enumerable.Range(1, CovariateTyping.MaxCategoricalLevels + 1).Select(i => (string?)i.ToString()).ToArray();
        Assert.Equal(CovariateKind.Categorical, CovariateTyping.Infer("Score", ten).Kind);
        Assert.Equal(CovariateKind.Numeric, CovariateTyping.Infer("Score", eleven).Kind);
    }

    [Fact]
    public void BlanksAndNaN_AreMissing_NotLevels()
    {
        var guess = CovariateTyping.Infer("Score", new string?[] { "1", "", null, "NaN", "2" });
        Assert.Equal(CovariateKind.Categorical, guess.Kind);
        Assert.Contains("2 distinct values", guess.Reason);
    }

    [Fact]
    public void NumbersAsCategories_KeepOneSpellingPerLevel()
    {
        var cov = Assert.IsType<CategoricalCovariate>(
            Covariate.FromMetadata("Patient", new string?[] { "1", "1.0", "2", " 2.0 ", null }));
        Assert.Equal(new string?[] { "1", "1", "2", "2", null }, cov.Values);
    }

    [Fact]
    public void AnExplicitType_Wins_AndSaysWhatItReplaced()
    {
        var ids = new string?[] { "1", "1", "2", "2", "3" };
        var cov = Covariate.FromMetadata("Patient", ids, CovariateKind.Numeric);
        Assert.Equal(CovariateKind.Numeric, cov.Kind);
        Assert.StartsWith("set explicitly; it would otherwise be categorical", cov.TypeReason);
        Assert.Equal("Patient (numeric)", cov.Describe());

        var inferred = Covariate.FromMetadata("Patient", ids);
        Assert.StartsWith("inferred: its name contains 'patient'", inferred.TypeReason);
    }

    [Fact]
    public void NumericCannotBeForcedOnText()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            Covariate.FromMetadata("Sex", new string?[] { "F", "M" }, CovariateKind.Numeric));
        Assert.Contains("'F' is not a number", ex.Message);
    }

    [Theory]
    [InlineData("numeric", CovariateKind.Numeric)]
    [InlineData(" Categorical ", CovariateKind.Categorical)]
    [InlineData("c", CovariateKind.Categorical)]
    public void Tokens(string token, CovariateKind kind)
    {
        Assert.Equal(kind, CovariateTyping.ParseToken(token));
        Assert.Equal(kind, CovariateTyping.ParseToken(CovariateTyping.Token(kind)));
    }

    [Fact]
    public void AnUnknownToken_IsNull() => Assert.Null(CovariateTyping.ParseToken("ordinal"));

    // ---- what the design does with each type ----

    // 12 samples: arm A = 0..5, arm B = 6..11, four subjects of three samples each, every subject in
    // one arm only - the shape of a sex contrast in a longitudinal cohort.
    private static readonly int[] A = { 0, 1, 2, 3, 4, 5 };
    private static readonly int[] B = { 6, 7, 8, 9, 10, 11 };
    private static readonly string?[] Patients = Repeat(("1", 3), ("2", 3), ("3", 3), ("4", 3));

    private static (double[,] Expr, string[] Ids) Matrix()
    {
        const int nF = 6;
        var expr = new double[nF, 12];
        for (var f = 0; f < nF; f++)
            for (var s = 0; s < 12; s++)
                expr[f, s] = 12 + f * 0.3 + ((f * 5 + s * 7) % 11) / 10.0 + (s >= 6 && f < 2 ? 1.5 : 0);
        return (expr, Enumerable.Range(0, nF).Select(f => $"F{f}").ToArray());
    }

    private static DifferentialOptions With(params Covariate[] covariates) => new()
    {
        Prior = VariancePrior.Global,
        Correction = MultipleTesting.None,
        Covariates = covariates,
    };

    /// <summary>
    /// The motivating case. An integer patient ID nested in the groups is categories, and as
    /// categories it would absorb the contrast - so it is dropped, by name, and the result is the
    /// unadjusted one. It never enters silently as one continuous column.
    /// </summary>
    [Fact]
    public void AnIntegerSubjectIdNestedInTheGroups_IsDroppedByName_NotFittedAsANumber()
    {
        var (expr, ids) = Matrix();
        var patient = Covariate.FromMetadata("Patient", Patients);

        var adjusted = Differential.Run(expr, ids, A, B, With(patient));
        var plain = Differential.Run(expr, ids, A, B, With());

        Assert.Equal(CovariateKind.Categorical, patient.Kind);
        Assert.Empty(adjusted.CovariatesUsed);
        Assert.Empty(adjusted.CovariatesAdjusted);
        Assert.Contains(adjusted.Messages, m => m.Contains("'Patient' is nested within the groups")
            && m.Contains("repeated samples"));
        foreach (var row in adjusted.Rows)
            Assert.Equal(plain.Rows.Single(r => r.FeatureId == row.FeatureId).T, row.T, 12);
    }

    /// <summary>The same column forced numeric is fitted as asked - with a warning on the result.</summary>
    [Fact]
    public void AnIdForcedNumeric_IsFitted_ButWarned()
    {
        var (expr, ids) = Matrix();
        var patient = Covariate.FromMetadata("Patient", Patients, CovariateKind.Numeric);

        var res = Differential.Run(expr, ids, A, B, With(patient));

        Assert.Equal(new[] { "Patient" }, res.CovariatesUsed);
        Assert.Equal(new[] { "Patient (numeric)" }, res.CovariatesAdjusted);
        Assert.Contains(res.Messages, m => m.Contains("'Patient' is fitted as numeric")
            && m.Contains("'patient'"));
    }

    /// <summary>A three-level integer batch spanning both arms enters as two indicators.</summary>
    [Fact]
    public void AThreeLevelIntegerBatch_IsTwoIndicators()
    {
        var (expr, ids) = Matrix();
        var batch = Covariate.FromMetadata("Batch", Repeat(("1", 2), ("2", 2), ("3", 2), ("1", 2), ("2", 2), ("3", 2)));

        var res = Differential.Run(expr, ids, A, B, With(batch));

        Assert.Equal(new[] { "Batch_2", "Batch_3" }, res.CovariatesUsed);
        Assert.Equal(new[] { "Batch (categorical)" }, res.CovariatesAdjusted);
        Assert.DoesNotContain(res.Messages, m => m.Contains("Batch"));
    }

    /// <summary>A covariate that spans the arms is not nested, even if it repeats within subjects.</summary>
    [Fact]
    public void ACategoricalSpanningBothArms_IsKept()
    {
        var (expr, ids) = Matrix();
        var site = Covariate.FromMetadata("Site", Repeat(("x", 3), ("y", 3), ("x", 3), ("z", 3)));

        var res = Differential.Run(expr, ids, A, B, With(site));

        Assert.Equal(new[] { "Site_y", "Site_z" }, res.CovariatesUsed);
        Assert.DoesNotContain(res.Messages, m => m.Contains("nested"));
    }

    /// <summary>
    /// The pane's per-covariate note is what the run would report for that covariate - the same
    /// builder, so the two cannot disagree - and says nothing for one that goes in as it is.
    /// </summary>
    [Fact]
    public void CovariateNotes_AreWhatTheRunReports()
    {
        var (expr, ids) = Matrix();
        var patient = Covariate.FromMetadata("Patient", Patients);
        var batch = Covariate.FromMetadata("Batch", Repeat(("1", 2), ("2", 2), ("3", 2), ("1", 2), ("2", 2), ("3", 2)));

        var run = Differential.Run(expr, ids, A, B, With(patient));
        Assert.Equal(run.Messages, Differential.CovariateNotes(patient, A, B, With()));
        Assert.Empty(Differential.CovariateNotes(batch, A, B, With()));

        // A trend has no arms, so nothing is nested in them.
        var x = Enumerable.Range(0, 12).Select(i => (double)(i % 4)).ToArray();
        Assert.DoesNotContain(Differential.CovariateNotes(patient, A.Concat(B).ToArray(), x, With() with
        {
            Design = DifferentialDesign.LinearTrend,
        }), m => m.Contains("nested"));
    }

    /// <summary>
    /// Under a paired design the preview takes the run's paired path, so a covariate the subject
    /// block makes redundant is reported there too - it used to be previewed as unpaired, with
    /// nothing to say, and then dropped by the run.
    /// </summary>
    [Fact]
    public void CovariateNotes_UnderPaired_SayWhatThePairedRunDoes()
    {
        var (expr, ids) = Matrix();
        // Subject k owns A[k] and B[k]; Sex is constant within each subject.
        var subjects = new string?[12];
        var sex = new string?[12];
        for (var k = 0; k < 6; k++)
        {
            subjects[A[k]] = subjects[B[k]] = $"S{k}";
            sex[A[k]] = sex[B[k]] = k % 2 == 0 ? "F" : "M";
        }

        var paired = With() with { Design = DifferentialDesign.Paired, SubjectLabels = subjects };
        var cov = Covariate.FromMetadata("Sex", sex);
        var run = Differential.Run(expr, ids, A, B, paired with { Covariates = new[] { cov } });
        var notes = Differential.CovariateNotes(cov, A, B, paired);

        Assert.Contains(notes, m => m.Contains("constant within each subject"));
        Assert.All(notes, m => Assert.Contains(m, run.Messages));
    }

    /// <summary>When a covariate's levels use up the residual df, the refusal names it and its column count.</summary>
    [Fact]
    public void RunningOutOfDf_NamesTheCovariate()
    {
        var (expr, ids) = Matrix();
        // Six levels spanning both arms: 5 indicators + intercept + group + ... leaves no residual df
        // alongside a second such covariate.
        var many = Covariate.FromMetadata("Batch", Repeat(("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1),
            ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1)), CovariateKind.Categorical);
        // Spans both arms too, so it is not dropped as nested and its 5 columns count.
        var more = Covariate.FromMetadata("Plate", Repeat(("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1), ("f", 1),
            ("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1), ("f", 1)));

        var ex = Assert.Throws<ArgumentException>(() => Differential.Run(expr, ids, A, B, With(many, more)));
        Assert.Contains("Batch (categorical) takes 5 columns", ex.Message);
        Assert.Contains("may be meant as numeric", ex.Message);
    }

    /// <summary>
    /// The case that found it: a numeric column holding #N/A for the QC pools (which have no age)
    /// was read as text, so it became categories.
    /// </summary>
    [Fact]
    public void MissingValueTokens_AreMissing_NotText()
    {
        var age = new string?[] { "#N/A", "40.7", "47", "47.5", "n/a", "NA", "null", "65.3", "70.8", "53.6", "58.9", "61.6" };
        Assert.Equal(CovariateKind.Numeric, CovariateTyping.Infer("Age at first visit", age).Kind);
        var cov = Assert.IsType<NumericCovariate>(Covariate.FromMetadata("Age at first visit", age));
        Assert.True(double.IsNaN(cov.Values[0]) && double.IsNaN(cov.Values[4]));
        // "None" is left alone: it is a real level often enough.
        Assert.Equal(CovariateKind.Categorical, CovariateTyping.Infer("Treatment", new string?[] { "None", "1", "2" }).Kind);
        Assert.False(CovariateTyping.IsMissing("None"));
    }

    [Fact]
    public void NonFiniteText_IsMissing()
    {
        var guess = CovariateTyping.Infer("Score", new string?[] { "NaN", "Infinity", null });
        Assert.Equal("it has no values", guess.Reason);

        var forced = Assert.IsType<NumericCovariate>(
            Covariate.FromMetadata("Score", new string?[] { "1.5", "Infinity", "-Infinity", "2.5" }, CovariateKind.Numeric));
        Assert.True(double.IsNaN(forced.Values[1]) && double.IsNaN(forced.Values[2]));

        var cat = Assert.IsType<CategoricalCovariate>(
            Covariate.FromMetadata("Batch", new string?[] { "1", "Infinity", "2" }, CovariateKind.Categorical));
        Assert.Null(cat.Values[1]);
    }

    [Fact]
    public void AConstantCategorical_IsSkippedAndSaid()
    {
        var (expr, ids) = Matrix();
        var res = Differential.Run(expr, ids, A, B,
            With(Covariate.FromMetadata("Run", Enumerable.Repeat((string?)"1", 12).ToArray())));

        Assert.Empty(res.CovariatesUsed);
        Assert.Contains("Covariate 'Run' is constant - skipped.", res.Messages);
    }

    /// <summary>
    /// Detection's covariates take an explicit type, and otherwise infer from the dataset's whole
    /// column - not from the subset of samples the detection matrix happens to hold.
    /// </summary>
    [Fact]
    public void DetectionCovariates_TakeTheExplicitType_OrInferFromTheWholeColumn()
    {
        var ds = DifferentialDataset.Load(Fixtures.Path2("mini", "e2e-sum", "output"), FeatureLevel.Protein);
        var col = ds.MetadataColumns.First(c => c == "batch");
        var someIds = ds.SampleIds.Take(3).ToArray();

        var inferred = DetectionAnalysis.CovariatesFor(ds, new[] { col }, someIds)!.Single();
        Assert.Equal(CovariateTyping.Infer(col, ds.MetadataValues(col)).Kind, inferred.Kind);

        var forced = DetectionAnalysis.CovariatesFor(ds, new[] { col }, someIds,
            new Dictionary<string, CovariateKind> { [col] = CovariateKind.Categorical })!.Single();
        Assert.Equal(CovariateKind.Categorical, forced.Kind);
    }
}
