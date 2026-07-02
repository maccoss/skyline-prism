using System;
using System.IO;
using System.IO.Compression;
using Microsoft.Data.Sqlite;
using SkylinePrism.Core.Library;
using Xunit;

namespace SkylinePrism.Tests.Library;

/// <summary>
/// Round-trips a synthetic Skyline .blib (SQLite) through SpectralLibrary.LoadBlib: exercises the
/// RefSpectra/RefSpectraPeaks join, both blob-decode paths (raw + zlib), base-peak normalization,
/// m/z matching, and the exact / I-L / modification-stripped GetSpectrum fallbacks - the BLIB reader
/// path that no fixture otherwise covers.
/// </summary>
public class BlibReaderTests
{
    private static byte[] Zlib(byte[] data)
    {
        using var outMs = new MemoryStream();
        using (var z = new ZLibStream(outMs, CompressionLevel.Optimal, leaveOpen: true))
            z.Write(data, 0, data.Length);
        var b = outMs.ToArray();
        // MaybeInflate only inflates blobs with the 0x78 0x9c zlib header (what Skyline/Bibliospec writes).
        Assert.True(b.Length >= 2 && b[0] == 0x78 && b[1] == 0x9c, $"unexpected zlib header {b[0]:X2} {b[1]:X2}");
        return b;
    }

    private static byte[] MzBlob(double[] mz, bool compress)
    {
        var raw = new byte[mz.Length * 8];
        Buffer.BlockCopy(mz, 0, raw, 0, raw.Length);
        return compress ? Zlib(raw) : raw;
    }

    private static byte[] IntensityBlob(float[] it, bool compress)
    {
        var raw = new byte[it.Length * 4];
        Buffer.BlockCopy(it, 0, raw, 0, raw.Length);
        return compress ? Zlib(raw) : raw;
    }

    private static void Exec(SqliteConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private static void Insert(
        SqliteConnection conn, int id, string pepSeq, string modSeq, int charge, double precMz,
        double[] mz, float[] it, bool compress)
    {
        using var s = conn.CreateCommand();
        s.CommandText = "INSERT INTO RefSpectra (id, peptideSeq, precursorMZ, precursorCharge, peptideModSeq) "
            + "VALUES ($id, $seq, $mz, $z, $mod)";
        s.Parameters.AddWithValue("$id", id);
        s.Parameters.AddWithValue("$seq", pepSeq);
        s.Parameters.AddWithValue("$mz", precMz);
        s.Parameters.AddWithValue("$z", charge);
        s.Parameters.AddWithValue("$mod", modSeq);
        s.ExecuteNonQuery();

        using var p = conn.CreateCommand();
        p.CommandText = "INSERT INTO RefSpectraPeaks (RefSpectraID, peakMZ, peakIntensity) VALUES ($id, $mz, $it)";
        p.Parameters.AddWithValue("$id", id);
        p.Parameters.AddWithValue("$mz", MzBlob(mz, compress));
        p.Parameters.AddWithValue("$it", IntensityBlob(it, compress));
        p.ExecuteNonQuery();
    }

    private static string WriteBlib()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_" + Guid.NewGuid().ToString("N") + ".blib");
        using (var conn = new SqliteConnection($"Data Source={path};Pooling=False"))
        {
            conn.Open();
            Exec(conn, "CREATE TABLE RefSpectra (id INTEGER PRIMARY KEY, peptideSeq TEXT, precursorMZ REAL, "
                + "precursorCharge INTEGER, peptideModSeq TEXT)");
            Exec(conn, "CREATE TABLE RefSpectraPeaks (RefSpectraID INTEGER, peakMZ BLOB, peakIntensity BLOB)");

            // A: zlib-compressed peaks; base peak (1000) at m/z 200.10.
            Insert(conn, 1, "PEPTIDEK", "PEPTIDEK", 2, 500.25,
                new[] { 100.05, 200.10, 300.15 }, new[] { 500f, 1000f, 250f }, compress: true);
            // B: uncompressed peaks; stored with I (query with L exercises the I/L fallback).
            Insert(conn, 2, "PEPTIDEIK", "PEPTIDEIK", 2, 480.0,
                new[] { 147.11, 260.20 }, new[] { 1000f, 500f }, compress: false);
            // C: modified sequence (query the stripped form exercises the stripped fallback).
            Insert(conn, 3, "PEPTIDECK", "PEPTIDEC(unimod:4)K", 2, 560.0,
                new[] { 175.10, 288.20 }, new[] { 800f, 400f }, compress: true);
        }
        return path;
    }

    [Fact]
    public void LoadBlib_DecodesPeaks_NormalizesBasePeak_AndMatchesByMz()
    {
        var path = WriteBlib();
        try
        {
            var lib = SpectralLibrary.LoadBlib(path);
            Assert.Equal(3, lib.Count);

            var a = lib.GetSpectrum("PEPTIDEK", 2);
            Assert.NotNull(a);
            Assert.Equal(3, a!.FragmentsByMz.Count);
            Assert.Equal(1.0, a.FragmentsByMz[200.10], 9); // base peak normalized to 1.0
            Assert.Equal(0.5, a.FragmentsByMz[100.05], 9);
            Assert.Equal(0.25, a.FragmentsByMz[300.15], 9);

            Assert.Equal(1.0, SpectralLibrary.MatchByMz(a, 200.11)!.Value, 9); // within 0.02 tolerance
            Assert.Null(SpectralLibrary.MatchByMz(a, 250.0));                  // no peak within tolerance
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void GetSpectrum_UncompressedAndFallbacks()
    {
        var path = WriteBlib();
        try
        {
            var lib = SpectralLibrary.LoadBlib(path);

            // Uncompressed blob decodes (spectrum B).
            Assert.NotNull(lib.GetSpectrum("PEPTIDEIK", 2));
            // I/L fallback: stored has I, query with L normalizes to it.
            Assert.NotNull(lib.GetSpectrum("PEPTIDELK", 2));
            // Modification-stripped fallback: query the unmodified form of the modified stored seq.
            Assert.NotNull(lib.GetSpectrum("PEPTIDECK", 2));
            // Genuine miss.
            Assert.Null(lib.GetSpectrum("NOSUCHPEPTIDE", 2));
            Assert.Null(lib.GetSpectrum("PEPTIDEK", 9)); // right seq, wrong charge
        }
        finally { File.Delete(path); }
    }
}
