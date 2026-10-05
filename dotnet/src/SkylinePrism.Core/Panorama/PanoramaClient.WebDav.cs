using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace SkylinePrism.Core.Panorama;

public sealed partial class PanoramaClient
{
    private static readonly HttpMethod Mkcol = new("MKCOL");

    /// <summary>A PUT that moves no bytes for this long is abandoned and retried.</summary>
    internal static TimeSpan UploadStallTimeout { get; set; } = TimeSpan.FromMinutes(2);

    /// <summary>The pause before retry n is n times this.</summary>
    internal static TimeSpan RetryDelayUnit { get; set; } = TimeSpan.FromSeconds(5);

    /// <summary>Attempts per file. There is no resume, so each retry restarts from byte zero.</summary>
    private const int MaxUploadAttempts = 3;

    /// <summary>Folders known to exist, so a batch into one folder does not MKCOL for every file.</summary>
    private readonly ConcurrentDictionary<string, byte> _knownFolders = new(StringComparer.Ordinal);

    /// <summary>Creates a folder in a file area, and any missing parents.</summary>
    /// <remarks>
    /// Panorama's MKCOL is single-level: a 409 means a parent is missing, so the parents are made
    /// first and this level is tried once more (measured by PanoramaBridge against panoramaweb.org).
    /// A folder that is already there answers 405 by PanoramaBridge's measurement and 200 by PRISM's
    /// (panoramaweb.org, October 2026); both mean what was wanted.
    /// </remarks>
    public async Task EnsureFolderAsync(string folder, CancellationToken cancellationToken = default)
    {
        var path = PanoramaPaths.ToFolder(folder);
        if (!PanoramaPaths.IsFileArea(path) || PanoramaPaths.Name(path).StartsWith('@') || _knownFolders.ContainsKey(path))
            return;

        var status = await TryMkcolAsync(path, cancellationToken).ConfigureAwait(false);
        if (status == HttpStatusCode.Conflict)
        {
            await EnsureFolderAsync(PanoramaPaths.Parent(path), cancellationToken).ConfigureAwait(false);
            status = await TryMkcolAsync(path, cancellationToken).ConfigureAwait(false);
        }

        if (status is HttpStatusCode.Created or HttpStatusCode.OK or HttpStatusCode.NoContent or HttpStatusCode.MethodNotAllowed)
        {
            _knownFolders.TryAdd(path, 0);
            return;
        }

        if (status == HttpStatusCode.Forbidden)
            throw new PanoramaException($"This Panorama account ({Credential}) may not create {path}.");
        throw new PanoramaException($"Panorama answered {(int)status} when creating the folder {path}.");
    }

    private async Task<HttpStatusCode> TryMkcolAsync(string folder, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(Mkcol, PanoramaPaths.Encode(PanoramaPaths.ToWebDav(folder)));
        using var response = await SendRawAsync(request, folder, MetadataTimeout, cancellationToken).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.Unauthorized)
            await EnsureSuccessAsync(response, folder, cancellationToken).ConfigureAwait(false);
        return response.StatusCode;
    }

    /// <summary>
    /// Uploads one file into a folder, replacing any file of that name, as a single streaming PUT.
    /// </summary>
    /// <remarks>
    /// <para>Panorama implements no <c>Content-Range</c> on PUT, so there is no partial or resumable
    /// upload: a file goes in one request whatever its size, guarded by a stall watchdog (no bytes
    /// moved for <see cref="UploadStallTimeout"/>) and retried from the start on a transport failure
    /// or a gateway's 502, 503 or 504.
    /// PUT answers 201 for a replacement as well as a creation.</para>
    /// <para>The file is opened sharing read, write and delete, so publishing never blocks the run or
    /// the GUI that may still be writing beside it (see "A reader must never take a file hostage" in
    /// CLAUDE.md).</para>
    /// </remarks>
    /// <param name="progress">Bytes sent so far in the current attempt; reset to 0 on a retry.</param>
    public async Task UploadFileAsync(
        string localPath, string folder, string name, IProgress<long>? progress = null,
        CancellationToken cancellationToken = default)
    {
        await EnsureFolderAsync(folder, cancellationToken).ConfigureAwait(false);
        var target = PanoramaPaths.Combine(folder, name);
        var url = PanoramaPaths.Encode(PanoramaPaths.ToWebDav(PanoramaPaths.ToFolder(folder))) + Uri.EscapeDataString(name);
        var length = new FileInfo(localPath).Length;

        for (var attempt = 1; ; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await using var file = new FileStream(localPath, FileMode.Open, FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete, bufferSize: 1, FileOptions.Asynchronous | FileOptions.SequentialScan);
            var content = new UploadContent(file, length, progress);

            using var request = new HttpRequestMessage(HttpMethod.Put, url) { Content = content };
            request.Headers.Authorization = Credential.ToAuthenticationHeader();
            using var abort = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            using var watchdog = new Timer(_ =>
            {
                try
                {
                    if (DateTime.UtcNow - content.LastProgressUtc > UploadStallTimeout)
                        abort.Cancel();
                }
                catch (ObjectDisposedException)
                {
                    // The attempt finished between the tick and the check; nothing to abandon.
                }
            }, null, TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(5));

            try
            {
                using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, abort.Token)
                    .ConfigureAwait(false);
                if ((int)response.StatusCode is 502 or 503 or 504)
                {
                    // The gateway in front of Panorama answering for a server that is briefly away: the
                    // same passing failure as a dropped connection, and retried the same way.
                    if (attempt >= MaxUploadAttempts)
                        throw new PanoramaException(
                            $"Uploading {name} to {PanoramaPaths.ToFolder(folder)} failed after {MaxUploadAttempts} attempts: the gateway "
                            + $"in front of Panorama answered {(int)response.StatusCode} {response.ReasonPhrase} each time. Panorama may be "
                            + "briefly down; publish again in a few minutes, and files already uploaded will be skipped.");
                    progress?.Report(0);
                    await Task.Delay(RetryDelayUnit * attempt, cancellationToken).ConfigureAwait(false);
                    continue;
                }

                await EnsureSuccessAsync(response, target, cancellationToken).ConfigureAwait(false);
                return;
            }
            catch (Exception ex) when (attempt < MaxUploadAttempts && !cancellationToken.IsCancellationRequested
                                       && ex is HttpRequestException or OperationCanceledException or IOException)
            {
                progress?.Report(0);
                await Task.Delay(RetryDelayUnit * attempt, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested
                                       && ex is HttpRequestException or OperationCanceledException or IOException)
            {
                throw new PanoramaException(
                    $"Uploading {name} to {PanoramaPaths.ToFolder(folder)} failed after {MaxUploadAttempts} attempts ({ex.Message}).", ex);
            }
        }
    }

    /// <summary>A file streamed in 1 MB writes, reporting each one, so a stall can be told from a slow link.</summary>
    private sealed class UploadContent : HttpContent
    {
        private const int ChunkBytes = 1 << 20;
        private readonly Stream _source;
        private readonly long _length;
        private readonly IProgress<long>? _progress;

        public UploadContent(Stream source, long length, IProgress<long>? progress)
        {
            _source = source;
            _length = length;
            _progress = progress;
            Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");
            Headers.ContentLength = length;
        }

        public DateTime LastProgressUtc { get; private set; } = DateTime.UtcNow;

        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context, CancellationToken cancellationToken)
        {
            var buffer = new byte[ChunkBytes];
            long sent = 0;
            int read;
            while ((read = await _source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await stream.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                sent += read;
                LastProgressUtc = DateTime.UtcNow;
                _progress?.Report(sent);
            }
        }

        protected override Task SerializeToStreamAsync(Stream stream, TransportContext? context) =>
            SerializeToStreamAsync(stream, context, CancellationToken.None);

        protected override bool TryComputeLength(out long length)
        {
            length = _length;
            return true;
        }
    }
}
