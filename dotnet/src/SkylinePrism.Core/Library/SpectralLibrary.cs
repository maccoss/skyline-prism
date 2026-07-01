using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
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
/// charge, mirroring spectral_library.BLIBLoader + SpectralLibraryRollup.get_spectrum: exact,
/// I/L-normalized, and modification-stripped key fallbacks.
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
        using var conn = new SqliteConnection($"Data Source={path};Mode=ReadOnly");
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

    /// <summary>Look up a spectrum with exact / I-L / stripped fallbacks (get_spectrum).</summary>
    public FragmentSpectrum? GetSpectrum(string modifiedSequence, int charge)
    {
        if (_byKey.TryGetValue(MakePeptideKey(modifiedSequence, charge), out var s))
            return s;

        var normKey = MakePeptideKey(NormalizeForMatching(modifiedSequence), charge);
        if (_byKey.TryGetValue(normKey, out s))
            return s;

        var strippedKey = MakeStrippedKey(modifiedSequence, charge);
        if (_strippedLookup.TryGetValue(strippedKey, out var origKey))
            return _byKey[origKey];

        var strippedNorm = strippedKey.Replace("L", "I");
        if (_strippedLookup.TryGetValue(strippedNorm, out origKey))
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
