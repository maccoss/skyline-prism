using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The mass arithmetic behind the theoretical claim set.
///
/// <para>The load-bearing test here is <see cref="EveryPrecursorInTheCohortFixtureReconciles"/>: it
/// checks PRISM's residue and modification tables against SKYLINE's, on every distinct precursor of
/// a committed real cohort. Hand-computed cases pin the arithmetic; only the fixture pins the
/// tables, and the tables are where a silent error would come from.</para>
/// </summary>
public class PeptideFragmentsTests
{
    private readonly ITestOutputHelper _out;

    public PeptideFragmentsTests(ITestOutputHelper o) => _out = o;

    /// <summary>
    /// Worked by hand from the monoisotopic residue masses, so the constants are pinned independently
    /// of the fixture below. LKGQAPPP at 2+ is 404.239795 in the Skyline export it came from.
    /// </summary>
    [Fact]
    public void NeutralMassAndPrecursorMzAreHandComputed()
    {
        // L + K + G + Q + A + P + P + P, plus water.
        const double expected = 113.08406396 + 128.09496301 + 57.02146372 + 128.05857750
            + 71.03711378 + (3 * 97.05276384) + PeptideFragments.Water;

        var mass = PeptideFragments.NeutralMass("LKGQAPPP");
        Assert.NotNull(mass);
        Assert.Equal(expected, mass!.Value, 8);

        var mz = PeptideFragments.PrecursorMz("LKGQAPPP", 2);
        Assert.NotNull(mz);
        Assert.Equal((expected + 2 * PeptideFragments.Proton) / 2, mz!.Value, 8);

        // ...and it agrees with what Skyline exported for that precursor.
        Assert.True(PeptideFragments.Reconciles("LKGQAPPP", 2, 404.239795));
    }

    /// <summary>Both notations Skyline and BLIB use carry the same mass.</summary>
    [Fact]
    public void UnimodAndBracketDeltaNotationsAgree()
    {
        var viaUnimod = PeptideFragments.NeutralMass("AC(unimod:4)K");
        var viaDelta = PeptideFragments.NeutralMass("AC[+57.021464]K");
        var unmodified = PeptideFragments.NeutralMass("ACK");

        Assert.NotNull(viaUnimod);
        Assert.NotNull(viaDelta);
        Assert.NotNull(unmodified);
        Assert.Equal(viaUnimod!.Value, viaDelta!.Value, 8);
        Assert.Equal(57.021464, viaUnimod.Value - unmodified!.Value, 6);
    }

    /// <summary>
    /// An N-terminal modification belongs to every b ion and to the precursor, and to no y ion. That
    /// falls out of folding it into the running prefix, which is exactly the kind of thing that is
    /// right by construction or wrong everywhere.
    /// </summary>
    [Fact]
    public void AnNTerminalModificationReachesBIonsAndNotYIons()
    {
        var plain = PeptideFragments.Enumerate("PEPTIDEK", 2);
        var acetyl = PeptideFragments.Enumerate("(unimod:1)PEPTIDEK", 2);

        double Mz(IReadOnlyList<TheoreticalIon> ions, IonSeries s, int ordinal, int z) =>
            ions.Single(i => i.Series == s && i.Ordinal == ordinal && i.Charge == z).Mz;

        // Compared with a TOLERANCE, not decimal places: Assert.Equal(expected, actual, digits)
        // ROUNDS both sides, and half this acetyl delta is 21.0052825 - exactly a rounding midpoint,
        // so the two sides round in opposite directions over a difference of 2e-14.
        const double tol = 1e-9;
        Assert.Equal(42.010565, Mz(acetyl, IonSeries.B, 3, 1) - Mz(plain, IonSeries.B, 3, 1), tol);
        Assert.Equal(0.0, Mz(acetyl, IonSeries.Y, 3, 1) - Mz(plain, IonSeries.Y, 3, 1), tol);
        Assert.Equal(
            42.010565 / 2,
            Mz(acetyl, IonSeries.Precursor, 0, 2) - Mz(plain, IonSeries.Precursor, 0, 2), tol);
    }

    /// <summary>
    /// b and y of the same peptide must sum back to the peptide: b_i + y_(n-i) = M + 2 protons, at
    /// 1+. If the prefix/suffix split is off by a residue this fails for every i.
    /// </summary>
    [Fact]
    public void ComplementaryBAndYIonsSumToThePrecursor()
    {
        const string seq = "SAMPLERPEPTIDEK";
        var mass = PeptideFragments.NeutralMass(seq)!.Value;
        var ions = PeptideFragments.Enumerate(seq, 2);
        var n = seq.Length;

        for (var i = 1; i < n; i++)
        {
            var b = ions.Single(x => x.Series == IonSeries.B && x.Ordinal == i && x.Charge == 1).Mz;
            var y = ions.Single(x => x.Series == IonSeries.Y && x.Ordinal == n - i && x.Charge == 1).Mz;
            Assert.Equal(mass + 2 * PeptideFragments.Proton, b + y, 6);
        }
    }

    /// <summary>
    /// 1+ and 2+, never above the precursor charge - a 2+ precursor has no 3+ fragment, and claiming
    /// m/z for one would claim another peptide's signal.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 2)]
    [InlineData(4, 2)]
    public void FragmentChargesAreCappedAtTwoAndAtThePrecursorCharge(int precursorCharge, int expectedMax)
    {
        var ions = PeptideFragments.Enumerate("PEPTIDEK", precursorCharge);
        var fragmentCharges = ions
            .Where(i => i.Series != IonSeries.Precursor)
            .Select(i => i.Charge)
            .Distinct()
            .OrderBy(z => z)
            .ToArray();

        Assert.Equal(Enumerable.Range(1, expectedMax).ToArray(), fragmentCharges);

        // The precursor is claimed at its own charge only, whatever that is.
        Assert.All(
            ions.Where(i => i.Series == IonSeries.Precursor),
            i => Assert.Equal(precursorCharge, i.Charge));
    }

    /// <summary>Three peaks by default, spaced by one C13, at the precursor's charge.</summary>
    [Fact]
    public void PrecursorIsotopesAreSpacedByOneNeutronOverTheCharge()
    {
        var ions = PeptideFragments.Enumerate("PEPTIDEK", 3)
            .Where(i => i.Series == IonSeries.Precursor)
            .OrderBy(i => i.Ordinal)
            .ToArray();

        Assert.Equal(3, ions.Length);
        Assert.Equal(PeptideFragments.IsotopeSpacing / 3, ions[1].Mz - ions[0].Mz, 8);
        Assert.Equal(PeptideFragments.IsotopeSpacing / 3, ions[2].Mz - ions[1].Mz, 8);
    }

    /// <summary>
    /// An unresolvable sequence yields null and an empty enumeration rather than a guess. This is the
    /// safety property the whole approach rests on: a wrong mass claims signal that is not the
    /// peptide's, and there is no conservative fallback.
    /// </summary>
    [Theory]
    [InlineData("PEPTIDEX")]              // X is not a residue
    [InlineData("PEPTIDEK(unimod:99999)")] // accession not in the table
    [InlineData("PEPT[Phospho]IDEK")]      // named modification carries no mass
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("peptidek")]               // lower case is not the export's notation
    public void AnUnresolvableSequenceIsNullAndClaimsNothing(string sequence)
    {
        Assert.Null(PeptideFragments.NeutralMass(sequence));
        Assert.Null(PeptideFragments.PrecursorMz(sequence, 2));
        Assert.Empty(PeptideFragments.Enumerate(sequence, 2));
        Assert.False(PeptideFragments.Reconciles(sequence, 2, 500.0));
    }

    /// <summary>A mass that does not match the export is refused, however well-formed the sequence.</summary>
    [Fact]
    public void ReconcilesRejectsAMassThatDisagreesWithTheExport()
    {
        // The true value; then the same precursor as if one carbamidomethyl had been missed.
        Assert.True(PeptideFragments.Reconciles("LKGQAPPP", 2, 404.239795));
        Assert.False(PeptideFragments.Reconciles("LKGQAPPP", 2, 404.239795 + (57.021464 / 2)));

        // Deamidation is the smallest delta in the table and still 1,000x the tolerance.
        Assert.False(PeptideFragments.Reconciles("LKGQAPPP", 2, 404.239795 + (0.984016 / 2)));
        Assert.False(PeptideFragments.Reconciles("LKGQAPPP", 0, 404.239795));
        Assert.False(PeptideFragments.Reconciles("LKGQAPPP", 2, double.NaN));
    }

    /// <summary>
    /// The one that pins the TABLES rather than the arithmetic: every distinct precursor of the
    /// committed cohort fixture, checked against the m/z Skyline itself computed for it.
    ///
    /// <para>This is an external reference - Skyline used its own residue and modification tables -
    /// so it catches a typo in <c>ResidueMass</c>, a wrong Unimod delta, and a notation misread, none
    /// of which any self-consistent test of this file could see.</para>
    /// </summary>
    [Fact]
    public void EveryPrecursorInTheCohortFixtureReconciles()
    {
        var dir = Path.Combine(AppContext.BaseDirectory, "fixtures", "cohort");
        Assert.True(Directory.Exists(dir), $"cohort fixture missing at {dir}");

        var precursors = CohortPrecursors(dir);
        Assert.True(precursors.Count > 100, $"expected the whole fixture, got {precursors.Count}");

        var worstPpm = 0.0;
        var unresolved = new List<string>();
        foreach (var (sequence, charge, skylineMz) in precursors)
        {
            var mz = PeptideFragments.PrecursorMz(sequence, charge);
            if (mz is null)
            {
                unresolved.Add($"{sequence} {charge}+");
                continue;
            }
            var ppm = (mz.Value - skylineMz) / skylineMz * 1e6;
            if (Math.Abs(ppm) > Math.Abs(worstPpm))
                worstPpm = ppm;
        }

        _out.WriteLine($"precursors checked : {precursors.Count:N0}");
        _out.WriteLine($"worst deviation    : {worstPpm:+0.0000;-0.0000} ppm");
        _out.WriteLine($"unresolved         : {unresolved.Count}");

        Assert.Empty(unresolved);
        Assert.True(
            Math.Abs(worstPpm) < 0.1,
            $"worst deviation from Skyline's own precursor m/z was {worstPpm:0.0000} ppm");
    }

    /// <summary>
    /// Distinct (modified sequence, charge, precursor m/z) from the partitioned fixture, read with
    /// DuckDB the way the pipeline reads it.
    /// </summary>
    private static List<(string Sequence, int Charge, double Mz)> CohortPrecursors(string dir)
    {
        var glob = Path.Combine(dir, "**", "*.parquet").Replace('\\', '/');
        var rows = new List<(string, int, double)>();
        using var conn = new DuckDB.NET.Data.DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "SELECT DISTINCT PeptideModifiedSequenceUnimodIds, PrecursorCharge, PrecursorMz "
            + $"FROM read_parquet('{glob}') "
            + "WHERE PeptideModifiedSequenceUnimodIds IS NOT NULL AND PrecursorMz IS NOT NULL";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            rows.Add((
                reader.GetString(0),
                Convert.ToInt32(reader.GetValue(1)),
                Convert.ToDouble(reader.GetValue(2))));
        }
        return rows;
    }

    /// <summary>
    /// Selenocysteine, with the selenium isotope convention pinned.
    ///
    /// <para>These three precursors are real - selenoproteins from a plasma EV cohort - and each one
    /// failed to reconcile before U was in the table, which is how it was found. The values are
    /// Skyline's own exported <c>Precursor Mz</c>, so this asserts PRISM against Skyline rather than
    /// against itself, and it fails if anyone "corrects" U to the lightest stable selenium isotope:
    /// Se-78 is 2 Da light, about 1,300 ppm here.</para>
    /// </summary>
    [Theory]
    [InlineData("AEENITESC(unimod:4)QUR", 2, 744.270341)]
    [InlineData("ENLPSLC(unimod:4)SUQGLR", 2, 762.822543)]
    [InlineData("TGSAITUQC(unimod:4)K", 2, 558.716484)]
    public void SelenocysteineReconcilesAgainstSkylinesOwnPrecursorMz(
        string sequence, int charge, double skylineMz)
    {
        Assert.True(
            PeptideFragments.Reconciles(sequence, charge, skylineMz),
            $"{sequence} {charge}+ computed {PeptideFragments.PrecursorMz(sequence, charge)}, "
            + $"Skyline exported {skylineMz}");

        Assert.NotEmpty(PeptideFragments.Enumerate(sequence, charge));
    }

    /// <summary>
    /// Pyrrolysine is NOT in the table, on purpose: nothing here exercises it and a guessed mass
    /// would be a peptide's worth of claims on m/z belonging to nothing. It must fail closed.
    /// </summary>
    [Fact]
    public void PyrrolysineIsUnresolvedRatherThanGuessed()
    {
        Assert.Null(PeptideFragments.NeutralMass("PEPTOIDEK"));
        Assert.Empty(PeptideFragments.Enumerate("PEPTOIDEK", 2));
    }

}
