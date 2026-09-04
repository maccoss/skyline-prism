#nullable enable

using System;
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

    // ONE extraction at a time in this process, whatever the destination.
    //
    // It used to be one per destination, which let the export loop run four at once - and that is
    // measurably the wrong thing, because an extraction is not CPU work, it is a single I/O path.
    // Measured on a Panorama folder over SMB: four concurrent extractions moved 3.7 MB/s each,
    // ~15 MB/s together, while ONE extraction from the same share to a local disk ran at 227 MB/s.
    // Concurrency there buys nothing and costs seeks; worse, it makes four archives all finish at the
    // end instead of one finishing early, so the run looks stalled for an hour.
    //
    // Deliberately not a cross-process lock: two PRISM windows extracting one archive at the same
    // moment would still collide, which is rare, announces itself as a sharing violation naming the
    // file, and is not prevented here.
    private static readonly SemaphoreSlim ExtractGate = new(1, 1);

    /// <summary>How often a long extraction reports progress. An hour of silence reads as a hang.</summary>
    private static readonly TimeSpan ProgressEvery = TimeSpan.FromSeconds(20);

    /// <summary>
    /// Extract <paramref name="archivePath"/> if it has not already been extracted, and return the
    /// path of the <c>.sky</c> inside. Reuses a previous extraction of the SAME archive - matched on
    /// its length and last-write time, the way the export cache is - because at ~17 GB a plate,
    /// re-extracting on every run is not a cost worth paying twice.
    /// </summary>
    /// <param name="fallbackDir">
    /// A directory PRISM may extract into when the archive's own folder is the wrong place - normally
    /// the run's output directory. Used only if it is on a local disk; see <see cref="ChooseTarget"/>.
    /// </param>
    public static string Extract(
        string archivePath, string? fallbackDir, Action<string>? log = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(archivePath))
            throw new FileNotFoundException($"Shared document archive not found: {archivePath}", archivePath);

        var archive = new FileInfo(archivePath);
        var target = ChooseTarget(archive, fallbackDir, log);

        // Outside the gate: a reusable extraction is a file check, and a run whose archives are all
        // already extracted must not queue up behind anything.
        if (TryReuse(archive, target, log) is { } reusedEarly)
            return reusedEarly;

        ExtractGate.Wait(cancellationToken);
        try
        {
            // Re-checked inside: another input may have extracted this same archive while we waited.
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
                + "later runs reuse it. One archive at a time: this is disk and network, not CPU.");

            var timer = System.Diagnostics.Stopwatch.StartNew();
            long done = 0;
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
                done += CopyEntry(entry, destination, bytes, done, timer, log, cancellationToken);
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
            log?.Invoke(
                $"Extracted {archive.Name} in {timer.Elapsed.TotalMinutes:N1} min "
                + $"({bytes / (1024.0 * 1024) / Math.Max(1, timer.Elapsed.TotalSeconds):N0} MB/s): {document}");
            return document;
        }
        finally
        {
            ExtractGate.Release();
        }
    }

    /// <summary>
    /// Copy one entry out, reporting progress while it goes. <c>ExtractToFile</c> would be shorter but
    /// says nothing until it finishes, and these entries are 12 GB - the whole complaint that prompted
    /// this was an extraction that "never seems to finish", from a log that had gone quiet.
    /// </summary>
    private static long CopyEntry(
        ZipArchiveEntry entry, string destination, long totalBytes, long alreadyDone,
        System.Diagnostics.Stopwatch timer, Action<string>? log, CancellationToken cancellationToken)
    {
        using var source = entry.Open();
        using var sink = new FileStream(
            destination, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);

        var buffer = new byte[1 << 20];
        long written = 0;
        var nextReport = timer.Elapsed + ProgressEvery;
        int read;
        while ((read = source.Read(buffer, 0, buffer.Length)) > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            sink.Write(buffer, 0, read);
            written += read;
            if (timer.Elapsed < nextReport)
                continue;
            nextReport = timer.Elapsed + ProgressEvery;
            var doneNow = alreadyDone + written;
            var rate = doneNow / (1024.0 * 1024) / Math.Max(1, timer.Elapsed.TotalSeconds);
            var remaining = rate > 0 ? (totalBytes - doneNow) / (1024.0 * 1024) / rate / 60 : 0;
            log?.Invoke(
                $"  {100.0 * doneNow / Math.Max(1, totalBytes):N0}% "
                + $"({doneNow / (1024.0 * 1024 * 1024):N1} of {totalBytes / (1024.0 * 1024 * 1024):N1} GB, "
                + $"{rate:N0} MB/s, about {remaining:N0} min left) - {Path.GetFileName(destination)}");
        }
        return written;
    }

    /// <summary>
    /// Where to extract. Beside the archive when the archive is on a LOCAL disk - which is where
    /// Skyline extracts too, so the result is a folder the user recognizes and can delete - and onto a
    /// local disk otherwise.
    ///
    /// <para><b>Why not always beside it.</b> A Panorama download folder is normally a network share,
    /// and extracting there reads ~12 GB and writes ~17 GB back over the same link. Measured on one:
    /// 3.7 MB/s per archive with four running, ~15 MB/s in total, against <b>227 MB/s</b> for the same
    /// archive extracted to a local disk - 15x. A 12-plate cohort is the difference between minutes
    /// and most of a day, and the first report of it was "it starts 4 files and never seems to fully
    /// uncompress". The extraction is a derived cache, not the user's data, so it does not belong on
    /// their share by default.</para>
    ///
    /// <para><see cref="ExtractDirEnvVar"/> overrides everything, which is what to reach for when the
    /// local disk is short of room - and <see cref="EnsureRoom"/> says so by name when it is.</para>
    /// </summary>
    private static string ChooseTarget(FileInfo archive, string? fallbackDir, Action<string>? log)
    {
        var stem = StemOf(archive.FullName);

        // BESIDE the archive, the stem alone identifies it: two archives in one folder cannot share a
        // name. Anywhere ELSE, they can - two Panorama folders both holding "Plate1.sky.zip" is an
        // ordinary thing - and a shared target directory is not a survivable collision. The stamp
        // does not save it: input A extracts, writes its stamp and hands the path to Skyline; input B
        // finds the stamp names a different archive, and re-extracts over the files A's Skyline is
        // reading. So a redirected target carries a digest of the archive's full path.
        var configured = Environment.GetEnvironmentVariable(ExtractDirEnvVar);
        if (!string.IsNullOrWhiteSpace(configured))
            return Path.Combine(configured!, Distinguish(stem, archive.FullName));

        // Beside the archive only when the archive is on a local disk AND that folder takes writes.
        if (IsOnLocalDisk(archive.DirectoryName)
            && CanWrite(Path.Combine(archive.DirectoryName!, ExtractRootName)))
        {
            return Path.Combine(archive.DirectoryName!, ExtractRootName, stem);
        }

        // Otherwise a local disk: the output directory if that is local, else the temp directory.
        // Distinguish() because these roots gather archives from anywhere, and two Panorama folders
        // can hold the same plate name.
        var localRoot = IsOnLocalDisk(fallbackDir) ? fallbackDir! : Path.GetTempPath();
        var target = Path.Combine(localRoot, ExtractRootName, Distinguish(stem, archive.FullName));
        log?.Invoke(
            $"{archive.Name} is not on a local disk, so it will be extracted to {target} rather than "
            + "beside it - extracting onto a network share was measured 15x slower. Set "
            + $"{ExtractDirEnvVar} to choose the disk yourself.");
        return target;
    }

    /// <summary>
    /// <c>&lt;stem&gt;-&lt;8 hex&gt;</c>: still recognizable, but one folder per ARCHIVE rather than per
    /// name. Only for targets that gather archives from several places; beside the archive the stem is
    /// already unique.
    /// </summary>
    private static string Distinguish(string stem, string archiveFullPath)
    {
        var hash = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(archiveFullPath.ToLowerInvariant()));
        return stem + "-" + Convert.ToHexString(hash, 0, 4).ToLowerInvariant();
    }

    /// <summary>
    /// Whether this path is on a local fixed disk. A mapped network drive, a UNC path and removable
    /// media are all "no" - the first because that is the case this exists to catch, the others
    /// because neither is somewhere to put 17 GB of derived files by default.
    /// </summary>
    internal static bool IsOnLocalDisk(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;
        try
        {
            var full = Path.GetFullPath(path!);
            if (full.StartsWith(@"\\", StringComparison.Ordinal))
                return false;   // UNC: DriveInfo cannot answer for it
            var root = Path.GetPathRoot(full);
            return !string.IsNullOrEmpty(root) && new DriveInfo(root).DriveType == DriveType.Fixed;
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException
                                       or UnauthorizedAccessException or IOException)
        {
            return false;   // unknowable: treat as not-local, which only costs a different target
        }
    }

    private static bool CanWrite(string dir)
    {
        try
        {
            Directory.CreateDirectory(dir);
            // A per-probe name: inputs are exported in parallel, so two archives in one folder probe
            // this directory at the same moment. A fixed name made them collide on each other's file
            // and one would conclude - wrongly, and with a misleading log line - that the folder was
            // not writable, then extract 17 GB somewhere else.
            var probe = Path.Combine(dir, ".prism-write-probe-" + Guid.NewGuid().ToString("N")[..8]);
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
        // The QUERY is best-effort and the DECISION is not, so they cannot share a try: free space is
        // unknowable on a UNC path (ArgumentException) and on a drive that is not ready - a
        // disconnected mapped drive throws DriveNotFoundException, an IOException. Catching IOException
        // around the whole thing would have swallowed this method's own refusal below, turning the
        // guard into the failure it exists to prevent.
        long free;
        string root;
        try
        {
            root = Path.GetPathRoot(Path.GetFullPath(target)) ?? "";
            if (string.IsNullOrEmpty(root))
                return;
            free = new DriveInfo(root).AvailableFreeSpace;
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException
                                       or UnauthorizedAccessException or IOException)
        {
            return;   // cannot tell: let the extraction try
        }

        // A margin, because the drive is doing other work and a document that only just fits is not a
        // document anyone can then process.
        if (free >= bytes + 1024L * 1024 * 1024)
            return;
        // Name the env var, not the output directory: the output directory is only used as a target
        // when it is itself on a local disk, so "point it somewhere bigger" was advice that would not
        // have worked for the case that gets here - a Panorama folder on a share.
        var roomier = RoomiestLocalDrive(bytes);
        throw new IOException(
            $"Extracting {archiveName} needs {bytes / (1024.0 * 1024 * 1024):N1} GB (plus a little "
            + $"headroom) but {root} has {free / (1024.0 * 1024 * 1024):N1} GB free. Set "
            + $"{ExtractDirEnvVar} to a drive with room"
            + (roomier is null ? "" : $" - {roomier} has the most free space")
            + ", or free some space.");
    }

    /// <summary>
    /// The local fixed drive with the most free space, when it could hold <paramref name="bytes"/> -
    /// so the out-of-space message can name somewhere that would actually work instead of leaving the
    /// user to go looking.
    /// </summary>
    private static string? RoomiestLocalDrive(long bytes)
    {
        try
        {
            return DriveInfo.GetDrives()
                .Where(d => d.DriveType == DriveType.Fixed && d.IsReady
                            && d.AvailableFreeSpace > bytes + 1024L * 1024 * 1024)
                .OrderByDescending(d => d.AvailableFreeSpace)
                .Select(d => $"{d.Name} ({d.AvailableFreeSpace / (1024.0 * 1024 * 1024):N0} GB free)")
                .FirstOrDefault();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
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
