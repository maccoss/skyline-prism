#nullable enable

using System;
using System.IO;

namespace SkylinePrism.Skyline;

/// <summary>
/// Validates that a file Skyline claims to have written is really parquet. Skyline picks the report
/// format from the output file's extension, and a failure can still leave a short/empty/CSV file behind,
/// so both export paths check the magic marker before trusting the result rather than assuming.
/// </summary>
public static class ParquetMagic
{
    /// <summary>A parquet file starts and ends with the 4-byte "PAR1" marker.</summary>
    public static bool IsValid(string path)
    {
        try
        {
            if (!File.Exists(path))
                return false;
            using var fs = File.OpenRead(path);
            if (fs.Length < 8)
                return false;
            var head = new byte[4];
            fs.ReadExactly(head);
            fs.Seek(-4, SeekOrigin.End);
            var tail = new byte[4];
            fs.ReadExactly(tail);
            ReadOnlySpan<byte> magic = "PAR1"u8;
            return head.AsSpan().SequenceEqual(magic) && tail.AsSpan().SequenceEqual(magic);
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }
}
