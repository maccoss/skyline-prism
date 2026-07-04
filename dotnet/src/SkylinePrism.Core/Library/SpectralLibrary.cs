using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.Data.Sqlite;

namespace SkylinePrism.Core.Library;

/// <summary>
/// A peptide's library fragment spectrum: relative fragment intensities keyed by product m/z
/// (rounded to 2 dp, base peak = 1.0). Ported from spectral_library.FragmentSpectrum
/// (BLIB path uses m/z matching only).
/// </summary>
public sealed class FragmentSpectrum
{
    public required string ModifiedSequence { get; init; }
    public int PrecursorCharge { get; init; }
    public double PrecursorMz { get; init; }
    public Dictionary<double, double> FragmentsByMz { get; } = new();
}

/// <summary>
/// Loads a Skyline BLIB (SQLite) spectral library and looks up spectra by modified sequence +
/// charge: exact match, then a modification-notation-insensitive fallback (Skyline unimod vs BLIB
/// mass-delta). I and L are kept distinct - they give different predicted spectra/RTs and each
/// detected peptide has its own exact predicted spectrum in the library, so I/L is never collapsed.
/// </summary>
public sealed class SpectralLibrary
{
    private readonly Dictionary<string, FragmentSpectrum> _byKey = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _strippedLookup = new(StringComparer.Ordinal);

    public int Count => _byKey.Count;

    public static string MakePeptideKey(string modifiedSequence, int charge) => $"{modifiedSequence}_{charge}";
    public static string NormalizeForMatching(string sequence) => sequence.Replace("L", "I");

    private static readonly Regex Unimod = new(@"\(unimod:\d+\)", RegexOptions.Compiled);
    private static readonly Regex MassDelta = new(@"\[[+-]?\d+\.?\d*\]", RegexOptions.Compiled);
    private static readonly Regex NamedMod = new(@"\[[^\]]+\]", RegexOptions.Compiled);
    private static readonly Regex ParenMod = new(@"\([^)]+\)", RegexOptions.Compiled);

    public static string StripModifications(string sequence)
    {
        var r = Unimod.Replace(sequence, "");
        r = MassDelta.Replace(r, "");
        r = NamedMod.Replace(r, "");
        r = ParenMod.Replace(r, "");
        return r;
    }

    public static string MakeStrippedKey(string modifiedSequence, int charge)
        => $"{StripModifications(modifiedSequence)}_{charge}";

    /// <summary>Load a .blib (SQLite) library. Only BLIB (m/z-based) matching is supported.</summary>
    public static SpectralLibrary LoadBlib(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("BLIB file not found", path);

        var lib = new SpectralLibrary();
        // Pooling=False so the library file handle is released as soon as we finish reading, rather
        // than lingering in the connection pool and locking the user's .blib.
        using var conn = new SqliteConnection($"Data Source={path};Mode=ReadOnly;Pooling=False");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "SELECT s.peptideSeq, s.precursorMZ, s.precursorCharge, s.peptideModSeq, " +
            "p.peakMZ, p.peakIntensity " +
            "FROM RefSpectra s JOIN RefSpectraPeaks p ON p.RefSpectraID = s.id";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var pepSeq = reader.IsDBNull(0) ? "" : reader.GetString(0);
            var precMz = reader.IsDBNull(1) ? 0.0 : reader.GetDouble(1);
            var precCharge = reader.IsDBNull(2) ? 0 : reader.GetInt32(2);
            var modSeq = reader.IsDBNull(3) ? "" : reader.GetString(3);
            var seq = string.IsNullOrEmpty(modSeq) ? pepSeq : modSeq;
            var key = MakePeptideKey(seq, precCharge);
            if (lib._byKey.ContainsKey(key))
                continue; // keep first

            if (reader.IsDBNull(4) || reader.IsDBNull(5))
                continue;
            var mz = DecodeMzBlob((byte[])reader.GetValue(4));
            var intensity = DecodeIntensityBlob((byte[])reader.GetValue(5));
            if (mz.Length != intensity.Length)
                continue;

            var spectrum = new FragmentSpectrum
            {
                ModifiedSequence = seq,
                PrecursorCharge = precCharge,
                PrecursorMz = precMz,
            };
            for (var i = 0; i < mz.Length; i++)
                spectrum.FragmentsByMz[Math.Round(mz[i], 2)] = intensity[i];

            lib._byKey[key] = spectrum;
            lib._strippedLookup.TryAdd(MakeStrippedKey(seq, precCharge), key);
        }
        return lib;
    }

    /// <summary>
    /// Load a spectral library, auto-detecting the format by extension: <c>.tsv</c> = Carafe/DIA-NN,
    /// <c>.blib</c> = Skyline BLIB. Mirrors Python <c>load_spectral_library</c>.
    /// </summary>
    public static SpectralLibrary Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("Spectral library not found", path);
        var suffix = Path.GetExtension(path).ToLowerInvariant();
        return suffix switch
        {
            ".tsv" => LoadCarafeTsv(path),
            ".blib" => LoadBlib(path),
            _ => throw new NotSupportedException(
                $"Unsupported spectral library format '{suffix}' ({path}). Use .blib (Skyline) or .tsv (Carafe/DIA-NN)."),
        };
    }

    /// <summary>
    /// Load a Carafe/DIA-NN TSV spectral library (ports <c>CarafeTSVLoader</c>). Required columns:
    /// ModifiedPeptide, PrecursorCharge, FragmentMz, RelativeIntensity. Decoys (Decoy != 0) are skipped;
    /// DIA-NN underscore-wrapped sequences (<c>_PEPTIDE_</c>) are unwrapped. Streams the file so 7GB+
    /// libraries load without materializing the text. Fragment intensities are stored by rounded m/z
    /// exactly as BLIB; the library-assist scale is invariant to the intensity normalization.
    /// </summary>
    public static SpectralLibrary LoadCarafeTsv(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("Carafe TSV library not found", path);

        var lib = new SpectralLibrary();
        using var reader = new StreamReader(path);
        var headerLine = reader.ReadLine();
        if (headerLine is null)
            throw new InvalidDataException($"Empty Carafe TSV library: {path}");

        var header = headerLine.Split('\t');
        var idx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < header.Length; i++)
            idx[header[i].Trim()] = i;
        int Col(string name) => idx.TryGetValue(name, out var i) ? i : -1;

        var iMod = Col("ModifiedPeptide");
        var iCharge = Col("PrecursorCharge");
        var iMz = Col("FragmentMz");
        var iIntensity = Col("RelativeIntensity");
        var iPrecMz = Col("PrecursorMz");
        var iDecoy = Col("Decoy");

        var missing = new[] { ("ModifiedPeptide", iMod), ("PrecursorCharge", iCharge),
            ("FragmentMz", iMz), ("RelativeIntensity", iIntensity) }
            .Where(c => c.Item2 < 0).Select(c => c.Item1).ToList();
        if (missing.Count > 0)
            throw new InvalidDataException(
                $"Carafe TSV missing required columns: {string.Join(", ", missing)} ({path}).");

        var maxIdx = Math.Max(Math.Max(iMod, iCharge), Math.Max(iMz, iIntensity));
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0)
                continue;
            var f = line.Split('\t');
            if (f.Length <= maxIdx)
                continue;
            if (iDecoy >= 0 && iDecoy < f.Length)
            {
                var d = f[iDecoy].Trim();
                if (d.Length > 0 && d != "0")
                    continue; // skip decoys (Decoy != 0)
            }

            var modSeq = f[iMod].Trim().Trim('_'); // DIA-NN wraps in underscores
            if (modSeq.Length == 0)
                continue;
            if (!int.TryParse(f[iCharge].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var charge))
                continue;
            if (!double.TryParse(f[iMz].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var mz))
                continue;
            if (!double.TryParse(f[iIntensity].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var intensity))
                continue;

            var key = MakePeptideKey(modSeq, charge);
            if (!lib._byKey.TryGetValue(key, out var spectrum))
            {
                double precMz = 0;
                if (iPrecMz >= 0 && iPrecMz < f.Length)
                    double.TryParse(f[iPrecMz].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out precMz);
                spectrum = new FragmentSpectrum
                {
                    ModifiedSequence = modSeq,
                    PrecursorCharge = charge,
                    PrecursorMz = precMz,
                };
                lib._byKey[key] = spectrum;
                lib._strippedLookup.TryAdd(MakeStrippedKey(modSeq, charge), key);
            }
            spectrum.FragmentsByMz[Math.Round(mz, 2)] = intensity; // rounded m/z key, same as BLIB
        }
        return lib;
    }

    /// <summary>
    /// Look up a spectrum by EXACT modified sequence + charge, then by a modification-notation-insensitive
    /// key (Skyline unimod vs BLIB mass-delta). I and L are kept distinct on purpose: they yield slightly
    /// different predicted spectra and RTs, and each detected peptide has its own exact predicted spectrum
    /// in the library (it was detected against it) - so we never collapse I/L to force a match.
    /// </summary>
    public FragmentSpectrum? GetSpectrum(string modifiedSequence, int charge)
    {
        if (_byKey.TryGetValue(MakePeptideKey(modifiedSequence, charge), out var s))
            return s;

        var strippedKey = MakeStrippedKey(modifiedSequence, charge);
        if (_strippedLookup.TryGetValue(strippedKey, out var origKey))
            return _byKey[origKey];

        return null;
    }

    /// <summary>Expected relative intensity for a transition, matched by product m/z within tolerance.</summary>
    public static double? MatchByMz(FragmentSpectrum spectrum, double productMz, double mzTolerance = 0.02)
    {
        var target = Math.Round(productMz, 2);
        double? best = null;
        var bestDiff = mzTolerance + 1;
        foreach (var (libMz, intensity) in spectrum.FragmentsByMz)
        {
            var diff = Math.Abs(libMz - target);
            if (diff < bestDiff && diff <= mzTolerance)
            {
                bestDiff = diff;
                best = intensity;
            }
        }
        return best;
    }

    private static double[] DecodeMzBlob(byte[] blob)
    {
        var data = MaybeInflate(blob);
        var n = data.Length / 8;
        var mz = new double[n];
        for (var i = 0; i < n; i++)
            mz[i] = BitConverter.ToDouble(data, i * 8);
        return mz;
    }

    private static double[] DecodeIntensityBlob(byte[] blob)
    {
        var data = MaybeInflate(blob);
        var n = data.Length / 4;
        var it = new double[n];
        double max = 0;
        for (var i = 0; i < n; i++)
        {
            it[i] = BitConverter.ToSingle(data, i * 4);
            if (it[i] > max)
                max = it[i];
        }
        if (max > 0)
            for (var i = 0; i < n; i++)
                it[i] /= max;
        return it;
    }

    private static byte[] MaybeInflate(byte[] blob)
    {
        if (blob.Length >= 2 && blob[0] == 0x78 && blob[1] == 0x9c)
        {
            using var ms = new MemoryStream(blob);
            using var z = new ZLibStream(ms, CompressionMode.Decompress);
            using var outMs = new MemoryStream();
            z.CopyTo(outMs);
            return outMs.ToArray();
        }
        return blob;
    }
}
