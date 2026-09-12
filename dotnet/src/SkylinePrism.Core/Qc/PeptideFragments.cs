using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

namespace SkylinePrism.Core.Qc;

/// <summary>Which series a theoretical ion belongs to.</summary>
public enum IonSeries
{
    /// <summary>N-terminal fragment.</summary>
    B,

    /// <summary>C-terminal fragment.</summary>
    Y,

    /// <summary>The intact precursor, surviving into the MS2 scan unfragmented.</summary>
    Precursor,
}

/// <param name="Mz">Observed m/z at <paramref name="Charge"/>.</param>
/// <param name="Series">Which series.</param>
/// <param name="Ordinal">Residues in the fragment; the isotope number for a precursor.</param>
/// <param name="Charge">Charge state this m/z is for.</param>
public readonly record struct TheoreticalIon(double Mz, IonSeries Series, int Ordinal, int Charge);

/// <summary>
/// Theoretical b/y fragments and precursor isotopes for a modified peptide sequence.
///
/// <para><b>Why this exists.</b> Ion accounting's assigned total is, by default, the signal in the
/// transitions the DOCUMENT carries - the fragments Skyline chose to quantify on, typically six per
/// precursor. That is the right numerator for "what is this quantification standing on", and the
/// wrong one for "how much of the acquired signal can this peptide account for at all": the
/// unfragmented precursor and the low-m/z fragments are poor quantifiers but real detector counts
/// that the peptide genuinely explains. This enumerates the second set, so both can be reported.</para>
///
/// <para><b>NOT YET WIRED UP, and there is a known gap to close first: HEAVY ISOTOPE LABELS.</b>
/// The PRISM report exports <c>Precursor.Peptide.ModifiedSequence.UnimodIds</c> - the PEPTIDE-level
/// sequence, which carries structural modifications only. An isotope label is a property of the
/// PRECURSOR (Skyline: <c>Precursor.IsotopeLabelType</c>, "indicating which isotope modifications
/// are applied"), so on a document with heavy internal standards this file computes the LIGHT mass,
/// fails to reconcile against the row's HEAVY precursor m/z, and drops the precursor. That is safe -
/// no wrong claim is made - but it would silently remove every heavy precursor from the total.
/// Closing it means exporting <c>Precursor.ModifiedSequenceUnimodIds</c> instead, which carries the
/// label AND its position; the position is why the delta cannot simply be derived from
/// <c>Precursor Mz</c>, since a +8 on the C-terminal K belongs to every y ion and no b ion. Whatever
/// wires this up must also report the reconciliation-failure COUNT per replicate, or the gap is
/// invisible in exactly the case it matters.</para>
///
/// <para><b>Every sequence is reconciled against Skyline's own precursor m/z before it is used</b>
/// (<see cref="Reconciles"/>). The residue table and the modification table here are PRISM's, not
/// Skyline's, so a modification this file does not know - or a residue it cannot parse - would
/// otherwise silently produce a peptide's worth of wrong m/z windows and claim signal that is not
/// the peptide's. Reconciling turns that into a peptide that is skipped and counted. The export
/// carries <c>Precursor Mz</c> for free, so the check costs nothing.</para>
/// </summary>
public static class PeptideFragments
{
    /// <summary>Mass of a proton, for converting a neutral mass to m/z.</summary>
    public const double Proton = 1.00727646688;

    /// <summary>Monoisotopic water, the neutral peptide's terminal contribution.</summary>
    public const double Water = 18.0105646863;

    /// <summary>C13 - C12, the spacing of the isotope envelope.</summary>
    public const double IsotopeSpacing = 1.00335483778;

    /// <summary>
    /// Monoisotopic residue masses. I and L are deliberately equal - they ARE equal in mass, which is
    /// why they cannot be told apart by m/z at all and why the library lookup refuses to collapse
    /// them (see <c>SpectralLibrary</c>).
    ///
    /// <para><b>U is selenocysteine</b>, the 21st amino acid, and its mass uses <b>Se-80</b> - the
    /// MOST ABUNDANT selenium isotope, not the lightest stable one. That is a real choice and it was
    /// not guessed: three selenoprotein precursors in a plasma EV cohort each give
    /// <c>150.95364</c> when the residue mass is solved for from Skyline's own exported
    /// <c>Precursor Mz</c>, agreeing to 1 uDa, and C3H5NOSe with Se-80 reproduces that exactly. Se-78
    /// would be 2.0 Da light - about 1,300 ppm on a 750 m/z doubly-charged precursor, so the two
    /// conventions are not close enough to confuse.</para>
    ///
    /// <para>O (pyrrolysine) is deliberately absent: no data here exercises it, and inventing a
    /// convention for it is exactly the guess <see cref="Reconciles"/> exists to prevent. A
    /// pyrrolysine peptide is therefore excluded and counted, which is the safe outcome.</para>
    /// </summary>
    private static readonly IReadOnlyDictionary<char, double> ResidueMass =
        new Dictionary<char, double>
        {
            ['G'] = 57.02146372, ['A'] = 71.03711378, ['S'] = 87.03202840, ['P'] = 97.05276384,
            ['V'] = 99.06841390, ['T'] = 101.04767846, ['C'] = 103.00918447, ['L'] = 113.08406396,
            ['I'] = 113.08406396, ['N'] = 114.04292744, ['D'] = 115.02694302, ['Q'] = 128.05857750,
            ['K'] = 128.09496301, ['E'] = 129.04259308, ['M'] = 131.04048508, ['H'] = 137.05891186,
            ['F'] = 147.06841390, ['R'] = 156.10111102, ['Y'] = 163.06332852, ['W'] = 186.07931294,
            ['U'] = 150.95363555,
        };

    /// <summary>
    /// Unimod accession to monoisotopic delta, for the notation Skyline exports
    /// (<c>C(unimod:4)</c>).
    ///
    /// <para>Deliberately a short list of what is actually seen rather than a vendored Unimod
    /// database. An accession missing here does not produce a wrong mass: the sequence fails to
    /// resolve, <see cref="Reconciles"/> rejects it, and the peptide is skipped and counted. Adding
    /// an entry is a one-line change, and the count tells you when one is needed.</para>
    /// </summary>
    private static readonly IReadOnlyDictionary<int, double> UnimodDelta =
        new Dictionary<int, double>
        {
            [1] = 42.010565,    // Acetyl
            [4] = 57.021464,    // Carbamidomethyl
            [5] = 43.005814,    // Carbamyl
            [6] = 58.005479,    // Carboxymethyl
            [7] = 0.984016,     // Deamidated
            [21] = 79.966331,   // Phospho
            [26] = 39.994915,   // Pyro-carbamidomethyl
            [27] = -18.010565,  // Glu->pyro-Glu
            [28] = -17.026549,  // Gln->pyro-Glu
            [34] = 14.015650,   // Methyl
            [35] = 15.994915,   // Oxidation
            [36] = 28.031300,   // Dimethyl
            [37] = 42.046950,   // Trimethyl
            [121] = 114.042927, // GG (ubiquitin remnant)
            [385] = -17.026549, // Ammonia-loss
        };

    // One residue and any modifications written after it. A modification group before the first
    // residue is an N-terminal modification and is matched separately.
    private static readonly Regex Token = new(
        @"([A-Z])((?:\(unimod:\d+\)|\[[+-]?[\d.]+\]|\[[A-Za-z][A-Za-z0-9_>-]*\])*)",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static readonly Regex ModGroup = new(
        @"\(unimod:(?<id>\d+)\)|\[(?<delta>[+-][\d.]+)\]|\[(?<name>[A-Za-z][A-Za-z0-9_>-]*)\]",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static readonly Regex LeadingMods = new(
        @"^(?:\(unimod:\d+\)|\[[+-]?[\d.]+\])+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// The neutral monoisotopic mass of <paramref name="modifiedSequence"/>, or null when any part
    /// of it cannot be resolved - an unknown residue, an unknown Unimod accession, or a
    /// modification written as a name rather than a mass.
    /// </summary>
    /// <remarks>
    /// Null rather than a best effort, and rather than a throw. A wrong mass here is not a wrong
    /// number in a report - it is a claim on m/z windows belonging to nothing, which would count
    /// another peptide's signal as this one's. There is no safe fallback, so the peptide is skipped.
    /// </remarks>
    public static double? NeutralMass(string? modifiedSequence)
    {
        if (string.IsNullOrWhiteSpace(modifiedSequence))
            return null;

        var seq = modifiedSequence.Trim();
        var total = Water;
        var at = 0;

        var nterm = LeadingMods.Match(seq);
        if (nterm.Success && nterm.Length > 0)
        {
            if (!AddMods(nterm.Value, ref total))
                return null;
            at = nterm.Length;
        }

        var residues = 0;
        while (at < seq.Length)
        {
            var m = Token.Match(seq, at);
            if (!m.Success || m.Index != at)
                return null;
            if (!ResidueMass.TryGetValue(m.Groups[1].Value[0], out var residue))
                return null;
            total += residue;
            residues++;
            if (!AddMods(m.Groups[2].Value, ref total))
                return null;
            at = m.Index + m.Length;
        }

        return residues > 0 ? total : null;
    }

    private static bool AddMods(string mods, ref double total)
    {
        if (mods.Length == 0)
            return true;
        var consumed = 0;
        foreach (Match m in ModGroup.Matches(mods))
        {
            consumed += m.Length;
            if (m.Groups["id"].Success)
            {
                if (!UnimodDelta.TryGetValue(
                        int.Parse(m.Groups["id"].Value, CultureInfo.InvariantCulture), out var d))
                    return false;
                total += d;
            }
            else if (m.Groups["delta"].Success)
            {
                total += double.Parse(m.Groups["delta"].Value, CultureInfo.InvariantCulture);
            }
            else
            {
                // A modification written as a NAME carries no mass. Resolving it would need the
                // very table this file deliberately does not vendor, so it is unresolvable.
                return false;
            }
        }
        return consumed == mods.Length;
    }

    /// <summary>The m/z of the intact peptide at <paramref name="charge"/>, or null.</summary>
    public static double? PrecursorMz(string? modifiedSequence, int charge)
    {
        if (charge <= 0)
            return null;
        var mass = NeutralMass(modifiedSequence);
        return mass is null ? null : (mass.Value + charge * Proton) / charge;
    }

    /// <summary>
    /// Whether this file's masses agree with the m/z Skyline exported for the same precursor.
    ///
    /// <para>The gate on every theoretical claim. It is an EXTERNAL check - Skyline computed that m/z
    /// from its own residue and modification tables - so it catches a missing Unimod accession, a
    /// residue table typo, and a sequence in a notation this file misreads, all of which would
    /// otherwise place claims on m/z that is not the peptide's.</para>
    /// </summary>
    /// <param name="tolerancePpm">
    /// Default 5 ppm, which is orders of magnitude looser than the agreement actually observed
    /// (worst 0.0022 ppm over the 385 distinct precursors of the committed cohort fixture) and far
    /// tighter than any real disagreement: a missing modification is off by a whole delta, and the
    /// smallest in the table above is deamidation at 0.984 Da - about 1,000 ppm on a 1,000 m/z
    /// precursor.
    /// </param>
    public static bool Reconciles(
        string? modifiedSequence, int charge, double skylinePrecursorMz, double tolerancePpm = 5.0)
    {
        if (!double.IsFinite(skylinePrecursorMz) || skylinePrecursorMz <= 0)
            return false;
        var mz = PrecursorMz(modifiedSequence, charge);
        if (mz is null)
            return false;
        return Math.Abs(mz.Value - skylinePrecursorMz) / skylinePrecursorMz * 1e6 <= tolerancePpm;
    }

    /// <summary>
    /// Every b and y fragment, plus the surviving precursor's isotopes, that this peptide can put
    /// into its own MS2 spectrum.
    ///
    /// <para><b>Fragment charges are 1+ and 2+, capped at the precursor charge.</b> A 2+ precursor
    /// cannot produce a 2+ fragment carrying the whole charge and a neutral partner, so capping is
    /// not a tuning choice; claiming m/z for charge states the precursor cannot support would claim
    /// signal belonging to something else.</para>
    ///
    /// <para><b>The precursor is claimed at its acquisition charge only</b>, over
    /// <paramref name="isotopes"/> + 1 peaks. Charge-reduced forms are deliberately absent: they are
    /// real in ETD-family activation and speculative for the HCD this is used on.</para>
    ///
    /// <para>Returns an empty list when the sequence does not resolve. Callers reconcile first, so an
    /// empty list here means the peptide contributes no theoretical claim rather than an empty
    /// one.</para>
    /// </summary>
    public static IReadOnlyList<TheoreticalIon> Enumerate(
        string? modifiedSequence, int precursorCharge, int isotopes = 2)
    {
        var empty = Array.Empty<TheoreticalIon>();
        if (precursorCharge <= 0 || string.IsNullOrWhiteSpace(modifiedSequence))
            return empty;

        var prefixes = PrefixMasses(modifiedSequence!);
        if (prefixes is null || prefixes.Count < 2)
            return empty;

        var n = prefixes.Count - 1;                 // residues
        var neutral = prefixes[n] + Water;          // the whole peptide
        var maxCharge = Math.Min(2, precursorCharge);
        var ions = new List<TheoreticalIon>((n - 1) * 2 * maxCharge + isotopes + 1);

        for (var z = 1; z <= maxCharge; z++)
        {
            for (var i = 1; i < n; i++)
            {
                // b_i is the N-terminal prefix as an acylium ion: the residues, no water.
                ions.Add(new TheoreticalIon(
                    (prefixes[i] + z * Proton) / z, IonSeries.B, i, z));

                // y_i is the C-terminal suffix plus water.
                var suffix = prefixes[n] - prefixes[n - i] + Water;
                ions.Add(new TheoreticalIon(
                    (suffix + z * Proton) / z, IonSeries.Y, i, z));
            }
        }

        for (var k = 0; k <= Math.Max(0, isotopes); k++)
        {
            ions.Add(new TheoreticalIon(
                (neutral + k * IsotopeSpacing + precursorCharge * Proton) / precursorCharge,
                IonSeries.Precursor, k, precursorCharge));
        }

        return ions;
    }

    /// <summary>
    /// Cumulative residue masses, index 0 = 0 and index i = the first i residues with their
    /// modifications. Null when the sequence does not resolve.
    /// </summary>
    /// <remarks>
    /// An N-terminal modification is folded into every prefix (it is present on b ions and on the
    /// precursor, absent from y ions), which falls out of starting the running total with it.
    /// </remarks>
    private static List<double>? PrefixMasses(string modifiedSequence)
    {
        var seq = modifiedSequence.Trim();
        var running = 0.0;
        var at = 0;

        var nterm = LeadingMods.Match(seq);
        if (nterm.Success && nterm.Length > 0)
        {
            if (!AddMods(nterm.Value, ref running))
                return null;
            at = nterm.Length;
        }

        var prefixes = new List<double> { running };
        while (at < seq.Length)
        {
            var m = Token.Match(seq, at);
            if (!m.Success || m.Index != at)
                return null;
            if (!ResidueMass.TryGetValue(m.Groups[1].Value[0], out var residue))
                return null;
            running += residue;
            if (!AddMods(m.Groups[2].Value, ref running))
                return null;
            prefixes.Add(running);
            at = m.Index + m.Length;
        }
        return prefixes;
    }
}
