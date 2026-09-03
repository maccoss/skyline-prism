#nullable enable

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using System.Threading;

namespace SkylinePrism.Skyline;

/// <summary>
/// A Skyline shared document archive - <c>.sky.zip</c> - which is how documents come off
/// PanoramaWeb: the <c>.sky</c>, its <c>.skyd</c> chromatogram cache, any spectral library and the
/// audit log in one file.
///
/// <para><b>Skyline's own command line cannot open one.</b> <c>--in</c> hands the path straight to an
/// XML reader (<c>CommandLine.OpenSkyFile</c>), and the pre-flight check passes a <c>.sky.zip</c> on
/// its extension alone, so the failure is a generic parse error whose text then contradicts itself:
/// <i>"does not appear to be a Skyline document. Skyline documents normally have a ".sky" or
/// ".sky.zip" filename extension"</i>. Verified against SkylineCmd on a real Panorama archive. Only
/// the GUI extracts (File &gt; Open, <c>SkylineWindow.OpenSharedFile</c>), so a document that is
/// already open in Skyline is a non-issue - it was extracted on the way in, and PRISM's live path
/// asks Skyline for the document path it ended up with.</para>
///
/// <para>So a closed archive has to be extracted before it can be exported from, which is what
/// <see cref="Extract"/> does. It is not cheap: a Panorama plate measured 13.7 GB compressed and
/// <b>17.4 GB extracted</b> (12.0 GB of that the <c>.skyd</c>, 4.4 GB the <c>.sky</c> itself), so the
/// extraction is reused across runs rather than repeated, and nothing selective is attempted - every
/// entry in those archives is a file the document needs.</para>
/// </summary>
public static class SharedDocumentArchive
{
    /// <summary>Skyline's <c>SrmDocumentSharing.EXT_SKY_ZIP</c>.</summary>
    public const string Extension = ".sky.zip";

    /// <summary>Where <see cref="Extract"/> puts its work: one folder per archive, under this.</summary>
    public const string ExtractRootName = "prism-extracted";

    /// <summary>
    /// Send every extraction under this directory instead of beside the archive. For a Panorama
    /// download folder on a slow share - writing 17 GB back over SMB is not free - or one the user
    /// would rather keep clean. The per-archive folder and the reuse rule are unchanged.
    /// </summary>
    public const string ExtractDirEnvVar = "PRISM_EXTRACT_DIR";

    /// <summary>
    /// Names the extraction PRISM made and the archive version it came from. Part of the on-disk
    /// layout - deleting it (or the folder) simply costs the next run an extraction.
    /// </summary>
    public const string StampFileName = ".prism-extract.json";

    /// <summary>Whether this path names a shared archive, by extension.</summary>
    public static bool IsArchive(string? path) =>
        !string.IsNullOrWhiteSpace(path)
        && path!.EndsWith(Extension, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// The archive's name without <c>.sky.zip</c> - the batch label and extraction folder name.
    /// <c>Path.GetFileNameWithoutExtension</c> is wrong here: it strips only <c>.zip</c> and leaves a
    /// trailing <c>.sky</c>, which would then show up in every sample ID as part of the batch.
    /// </summary>
    public static string StemOf(string path)
    {
        var name = Path.GetFileName(path);
        return IsArchive(name) ? name[..^Extension.Length] : Path.GetFileNameWithoutExtension(name);
    }

    /// <summary>
    /// Open the archive's <c>.sky</c> entry for reading, WITHOUT extracting anything. The returned
    /// stream owns the archive, so disposing it closes both.
    ///
    /// <para>This is what lets the tool show a document's replicates, annotations, enzyme and
    /// extraction tolerance the moment an archive is added: the header parsers stop at
    /// <c>&lt;/settings_summary&gt;</c>, so only the first part of the entry is decompressed - worth
    /// having when the entry itself is 4.4 GB.</para>
    /// </summary>
    /// <exception cref="InvalidDataException">The zip holds no <c>.sky</c>, or more than one.</exception>
    public static Stream OpenDocumentEntry(string archivePath)
    {
        var archive = ZipFile.OpenRead(archivePath);
        try
        {
            var entry = FindDocumentEntry(archive, archivePath);
            return new OwnedEntryStream(archive, entry.Open());
        }
        catch
        {
            archive.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The UNCOMPRESSED size of the archive's <c>.sky</c> entry, read from the central directory (so it
    /// costs nothing), or 0 when the archive cannot be read.
    ///
    /// <para>This is the number the export memory budget wants, not the archive's own length: the
    /// budget models a headless Skyline at roughly twice the <c>.sky</c>, and on a measured Panorama
    /// plate the archive is 13.7 GB while the document inside it is 4.4 GB. Sizing off the archive
    /// over-estimates by ~3x, which is safe but can drop a nine-plate cohort to one export at a
    /// time for no reason.</para>
    /// </summary>
    public static long DocumentBytes(string archivePath)
    {
        try
        {
            using var archive = ZipFile.OpenRead(archivePath);
            return FindDocumentEntry(archive, archivePath).Length;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return 0;
        }
    }

    /// <summary>
    /// The single <c>.sky</c> entry, by Skyline's own rule (<c>SrmDocumentSharing.FindSharedSkylineFile</c>):
    /// among the entries at or near the top level, exactly one must end in <c>.sky</c>. Deeper entries
    /// are ignored because a shared archive may carry vendor data folders.
    /// </summary>
    private static ZipArchiveEntry FindDocumentEntry(ZipArchive archive, string archivePath)
    {
        var candidates = archive.Entries
            .Where(e => e.FullName.EndsWith(".sky", StringComparison.OrdinalIgnoreCase)
                        && e.FullName.Count(c => c is '/' or '\\') <= 2)
            .ToList();

        if (candidates.Count == 1)
            return candidates[0];

        var name = Path.GetFileName(archivePath);
        throw new InvalidDataException(
            candidates.Count == 0
                ? $"'{name}' contains no Skyline document (.sky), so it is not a shared document "
                  + "archive. If it came from Panorama, download the document rather than a folder of "
                  + "files."
                : $"'{name}' contains {candidates.Count} Skyline documents "
                  + $"({string.Join(", ", candidates.Take(4).Select(c => c.FullName))}), so PRISM "
                  + "cannot tell which one to use. Extract it and add the document you want.");
    }

    /// <summary>What a previous extraction produced, and from what.</summary>
    private sealed record ExtractStamp(
        string Archive, long Length, long LastWriteUtcTicks, string DocumentEntry, string Tool);

    private static string ToolVersion =>
        typeof(SharedDocumentArchive).Assembly.GetName().Version?.ToString() ?? "0";

    // One extraction at a time per destination. Two inputs can name the same archive (the same plate
    // added twice, or two runs sharing a download folder), and the export loop runs inputs in
    // parallel - so without this they would extract over each other's files.
    private static readonly ConcurrentDictionary<string, object> Locks =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Extract <paramref name="archivePath"/> if it has not already been extracted, and return the
    /// path of the <c>.sky</c> inside. Reuses a previous extraction of the SAME archive - matched on
    /// its length and last-write time, the way the export cache is - because at ~17 GB a plate,
    /// re-extracting on every run is not a cost worth paying twice.
    /// </summary>
    /// <param name="fallbackDir">
    /// Where to extract when the archive's own folder cannot be written to - a read-only share, or a
    /// Panorama download directory the user would rather keep clean. Normally the run's output
    /// directory.
    /// </param>
    public static string Extract(
        string archivePath, string? fallbackDir, Action<string>? log = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(archivePath))
            throw new FileNotFoundException($"Shared document archive not found: {archivePath}", archivePath);

        var archive = new FileInfo(archivePath);
        var target = ChooseTarget(archive, fallbackDir, log);

        lock (Locks.GetOrAdd(target, _ => new object()))
        {
            if (TryReuse(archive, target, log) is { } reused)
                return reused;

            using var zip = ZipFile.OpenRead(archivePath);
            var entryName = FindDocumentEntry(zip, archivePath).FullName;
            var bytes = zip.Entries.Sum(e => e.Length);

            EnsureRoom(target, bytes, archive.Name);
            Directory.CreateDirectory(target);
            log?.Invoke(
                $"Extracting {archive.Name} ({archive.Length / (1024.0 * 1024 * 1024):N1} GB compressed, "
                + $"{bytes / (1024.0 * 1024 * 1024):N1} GB extracted, {zip.Entries.Count} file(s)) to "
                + $"{target}. Skyline's command line cannot open a .sky.zip, so this happens once - "
                + "later runs reuse it.");

            foreach (var entry in zip.Entries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var destination = ResolveEntryPath(target, entry.FullName);
                if (entry.FullName.EndsWith('/') || entry.FullName.EndsWith('\\'))
                {
                    Directory.CreateDirectory(destination);
                    continue;
                }
                var parent = Path.GetDirectoryName(destination);
                if (!string.IsNullOrEmpty(parent))
                    Directory.CreateDirectory(parent);
                // Overwrite, so a run killed mid-extraction is repaired rather than refused. The
                // stamp is written last, so that half-done state is never mistaken for reusable.
                entry.ExtractToFile(destination, overwrite: true);
            }

            var document = ResolveEntryPath(target, entryName);
            if (!File.Exists(document))
            {
                throw new InvalidDataException(
                    $"Extracted {archive.Name} but '{entryName}' is not there. The archive may be "
                    + "truncated - if it is still downloading, wait for it to finish.");
            }

            File.WriteAllText(
                Path.Combine(target, StampFileName),
                JsonSerializer.Serialize(new ExtractStamp(
                    archive.FullName, archive.Length, archive.LastWriteTimeUtc.Ticks, entryName,
                    ToolVersion)));
            log?.Invoke($"Extracted {archive.Name}: {document}");
            return document;
        }
    }

    /// <summary>
    /// Beside the archive by default - which is where Skyline extracts too, so the result is a folder
    /// the user recognizes and can delete to reclaim the space - falling back to the run's output
    /// directory when that is not writable.
    /// </summary>
    private static string ChooseTarget(FileInfo archive, string? fallbackDir, Action<string>? log)
    {
        var stem = StemOf(archive.FullName);

        var configured = Environment.GetEnvironmentVariable(ExtractDirEnvVar);
        if (!string.IsNullOrWhiteSpace(configured))
            return Path.Combine(configured!, stem);

        var beside = archive.DirectoryName is null
            ? null
            : Path.Combine(archive.DirectoryName, ExtractRootName, stem);

        if (beside is not null && CanWrite(Path.Combine(archive.DirectoryName!, ExtractRootName)))
            return beside;

        if (string.IsNullOrWhiteSpace(fallbackDir))
        {
            return beside
                ?? Path.Combine(Path.GetTempPath(), ExtractRootName, stem);
        }

        log?.Invoke(
            $"Cannot write beside {archive.Name}, so it will be extracted under {fallbackDir} instead.");
        return Path.Combine(fallbackDir!, ExtractRootName, stem);
    }

    private static bool CanWrite(string dir)
    {
        try
        {
            Directory.CreateDirectory(dir);
            var probe = Path.Combine(dir, ".prism-write-probe");
            File.WriteAllBytes(probe, Array.Empty<byte>());
            File.Delete(probe);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            return false;
        }
    }

    /// <summary>The extracted document from a previous run of the SAME archive, or null.</summary>
    private static string? TryReuse(FileInfo archive, string target, Action<string>? log)
    {
        try
        {
            var stampPath = Path.Combine(target, StampFileName);
            if (!File.Exists(stampPath))
                return null;
            var stamp = JsonSerializer.Deserialize<ExtractStamp>(File.ReadAllText(stampPath));
            if (stamp is null
                || !string.Equals(stamp.Archive, archive.FullName, StringComparison.OrdinalIgnoreCase)
                || stamp.Length != archive.Length
                || stamp.LastWriteUtcTicks != archive.LastWriteTimeUtc.Ticks
                || string.IsNullOrEmpty(stamp.DocumentEntry))
            {
                return null;
            }

            var document = ResolveEntryPath(target, stamp.DocumentEntry);
            if (!File.Exists(document) || new FileInfo(document).Length == 0)
                return null;

            log?.Invoke(
                $"Reusing the extracted {archive.Name} at {document} (the archive is unchanged since "
                + "it was extracted).");
            return document;
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return null;   // a bad stamp costs an extraction, not a run
        }
    }

    /// <summary>
    /// An entry's destination, rejecting any name that would escape <paramref name="target"/>. A zip
    /// is user data and an entry name is not a trusted path: <c>../</c> segments and rooted names are
    /// the standard zip-slip trick, and these archives arrive over the network from a server.
    /// </summary>
    private static string ResolveEntryPath(string target, string entryName)
    {
        var root = Path.GetFullPath(target);
        var combined = Path.GetFullPath(Path.Combine(root, entryName.Replace('/', Path.DirectorySeparatorChar)));
        if (!combined.StartsWith(
                root.EndsWith(Path.DirectorySeparatorChar) ? root : root + Path.DirectorySeparatorChar,
                StringComparison.OrdinalIgnoreCase)
            && !string.Equals(combined, root, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Entry '{entryName}' would be extracted outside {target}, so the archive is refused.");
        }
        return combined;
    }

    /// <summary>
    /// Refuse before starting rather than fill the disk and fail part-way through. 17 GB per plate is
    /// enough that "is there room" is a real question, and a half-extracted document is worse than a
    /// clear message.
    /// </summary>
    private static void EnsureRoom(string target, long bytes, string archiveName)
    {
        try
        {
            var root = Path.GetPathRoot(Path.GetFullPath(target));
            if (string.IsNullOrEmpty(root))
                return;
            var free = new DriveInfo(root).AvailableFreeSpace;
            // A margin, because the drive is doing other work and a document that only just fits is
            // not a document anyone can then process.
            var needed = bytes + 1024L * 1024 * 1024;
            if (free >= needed)
                return;
            throw new IOException(
                $"Extracting {archiveName} needs {bytes / (1024.0 * 1024 * 1024):N1} GB (plus a little "
                + $"headroom) but {root} has {free / (1024.0 * 1024 * 1024):N1} GB free. Free some "
                + "space, or point the run's output directory at a bigger drive.");
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or UnauthorizedAccessException)
        {
            // Unknowable free space (a UNC path, an odd mount): let the extraction try.
        }
    }

    /// <summary>A zip entry's stream that owns its archive, so one <c>using</c> closes both.</summary>
    private sealed class OwnedEntryStream : Stream
    {
        private readonly ZipArchive _archive;
        private readonly Stream _inner;

        public OwnedEntryStream(ZipArchive archive, Stream inner)
        {
            _archive = archive;
            _inner = inner;
        }

        public override bool CanRead => _inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => _inner.Position;
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        public override int Read(Span<byte> buffer) => _inner.Read(buffer);
        public override void Flush() => _inner.Flush();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _inner.Dispose();
                _archive.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
