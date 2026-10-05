using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.Pipeline;

namespace SkylinePrism.Core.Panorama;

/// <summary>A folder or file in a Panorama listing.</summary>
/// <param name="Folder">Its folder path, for example <c>/MacCoss/maccoss/@files/RawFiles</c>.</param>
/// <param name="Length">Bytes, for a file; 0 for a folder.</param>
/// <param name="CanUpload">Whether this account may write into it (a folder) - false when the server did not say.</param>
public sealed record PanoramaEntry(
    string Name, string Folder, bool IsFolder, long Length, DateTimeOffset? LastModifiedUtc, bool CanUpload);

/// <summary>A Panorama request failed; the message is written for the person using PRISM.</summary>
public sealed class PanoramaException(string message, Exception? inner = null) : Exception(message, inner)
{
    /// <summary>Panorama did not accept the sign-in, so asking for another one may help.</summary>
    public bool IsSignInProblem { get; init; }
}

/// <summary>
/// Talks to Panorama (LabKey Server), signed in with HTTP Basic on every request. Folders are
/// listed the way PanoramaBridge browses them, one level at a time with LabKey's WebDAV listing
/// (<c>GET /_webdav/...?method=json</c>); files are written with WebDAV (PanoramaClient.WebDav.cs);
/// wiki pages and their attachments with LabKey's wiki API (PanoramaClient.Wiki.cs).
/// </summary>
/// <remarks>
/// <para>Ported from LabOps's <c>PanoramaClient</c> (which LabOps built on PanoramaBridge's
/// verified server behavior), MacCoss Lab. LabOps only reads Panorama and writes one wiki page;
/// PRISM adds the WebDAV upload and wiki attachments, both of which follow PanoramaBridge's
/// measurements of what panoramaweb.org actually does rather than what the specifications say.</para>
/// <para><b>No redirects and no cookies.</b> A redirect means a sign-in page, which must fail rather
/// than look like an empty folder; stateless Basic needs no session. The one place a session is
/// needed - LabKey's CSRF token for a wiki POST - fetches it for that request alone.</para>
/// </remarks>
public sealed partial class PanoramaClient : IDisposable
{
    private static readonly TimeSpan MetadataTimeout = TimeSpan.FromSeconds(60);

    private readonly HttpClient _http;

    public PanoramaClient(Uri server, PanoramaCredential credential, HttpMessageHandler? handler = null)
    {
        Server = server;
        Credential = credential;
        // A handler passed in belongs to the caller (a test's fake serves several clients), so it is
        // not disposed with this one.
        _http = new HttpClient(handler ?? new SocketsHttpHandler { AllowAutoRedirect = false, UseCookies = false },
            disposeHandler: handler is null)
        {
            BaseAddress = server,
            // Uploads run for minutes; each request sets its own limit instead.
            Timeout = Timeout.InfiniteTimeSpan,
        };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd($"Skyline-PRISM/{PrismVersion.Current}");
    }

    public Uri Server { get; }

    /// <summary>The sign-in in use - its <see cref="PanoramaCredential.Source"/> names it in messages.</summary>
    public PanoramaCredential Credential { get; }

    /// <summary>Lists one folder: its subfolders and files.</summary>
    public async Task<IReadOnlyList<PanoramaEntry>> ListAsync(string folder, CancellationToken cancellationToken = default)
    {
        var body = await GetAsync(PanoramaPaths.Encode(PanoramaPaths.ToWebDav(folder)) + "?method=json",
            PanoramaPaths.ToFolder(folder), cancellationToken).ConfigureAwait(false);
        return ParseListing(body!, folder);
    }

    /// <summary>
    /// Whether Panorama accepts this sign-in: lists the projects, the cheapest request that needs it.
    /// Null when it does, else why not.
    /// </summary>
    public async Task<PanoramaException?> CheckSignInAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            await ListAsync("/", cancellationToken).ConfigureAwait(false);
            return null;
        }
        catch (PanoramaException ex)
        {
            return ex;
        }
    }

    /// <summary>
    /// The server's MD5 of every file directly inside a folder (LabKey's <c>?method=md5sum</c>), by
    /// name; empty when the folder does not exist yet.
    /// </summary>
    /// <remarks>
    /// The hash is computed by the server over the bytes it stored, which is what makes it worth
    /// asking for: a matching hash proves the upload landed intact, and lets a republish skip a
    /// file that is already there. Subfolders are not included - one request per folder.
    /// </remarks>
    public async Task<IReadOnlyDictionary<string, string>> Md5sAsync(string folder, CancellationToken cancellationToken = default)
    {
        var body = await SendAsync(
            () => new HttpRequestMessage(HttpMethod.Get, PanoramaPaths.Encode(PanoramaPaths.ToWebDav(folder)) + "?method=md5sum"),
            PanoramaPaths.ToFolder(folder), allowMissing: true, MetadataTimeout, cancellationToken).ConfigureAwait(false);
        return body is null ? new Dictionary<string, string>(StringComparer.Ordinal) : ParseMd5s(body);
    }

    private Task<string?> GetAsync(string relativeUrl, string display, CancellationToken cancellationToken) =>
        SendAsync(() => new HttpRequestMessage(HttpMethod.Get, relativeUrl), display, allowMissing: false,
            MetadataTimeout, cancellationToken);

    /// <summary>
    /// Sends a request and returns the answer as text, with every failure turned into a message for
    /// the person; null for a 404 when <paramref name="allowMissing"/>.
    /// </summary>
    private async Task<string?> SendAsync(
        Func<HttpRequestMessage> build, string display, bool allowMissing, TimeSpan limit,
        CancellationToken cancellationToken)
    {
        using var request = build();
        using var response = await SendRawAsync(request, display, limit, cancellationToken).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.NotFound && allowMissing)
            return null;
        await EnsureSuccessAsync(response, display, cancellationToken).ConfigureAwait(false);
        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task<HttpResponseMessage> SendRawAsync(
        HttpRequestMessage request, string display, TimeSpan limit, CancellationToken cancellationToken)
    {
        request.Headers.Authorization = Credential.ToAuthenticationHeader();
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(limit);
        try
        {
            return await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is HttpRequestException
                                   || (ex is OperationCanceledException && !cancellationToken.IsCancellationRequested))
        {
            throw new PanoramaException(
                $"Could not reach Panorama for {display} ({ex.Message}). Check the network connection.", ex);
        }
    }

    /// <summary>Every failure status as a sentence a person can act on.</summary>
    private async Task EnsureSuccessAsync(HttpResponseMessage response, string display, CancellationToken cancellationToken)
    {
        if (response.StatusCode == HttpStatusCode.Unauthorized || (int)response.StatusCode is >= 300 and < 400)
        {
            var which = Credential.Source == "typed"
                ? Credential.IsApiKey ? "this API key" : "this user name and password"
                : $"the sign-in from {Credential.Source} ({Credential})";
            throw new PanoramaException(
                $"Panorama did not accept {which}. "
                + (Credential.Source == "typed" ? "Check it and try again." : "The API key may have expired, or the password changed."))
            {
                IsSignInProblem = true,
            };
        }

        if (response.StatusCode == HttpStatusCode.Forbidden)
            throw new PanoramaException($"This Panorama account ({Credential}) may not change {display}.");

        if (response.StatusCode == HttpStatusCode.NotFound)
            throw new PanoramaException($"{display} is not on Panorama.");

        if (!response.IsSuccessStatusCode)
        {
            // LabKey's APIs explain a refusal in JSON: {"exception": "..."}.
            var detail = ApiError(await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
            throw new PanoramaException(detail is null
                ? $"Panorama answered {(int)response.StatusCode} {response.ReasonPhrase} for {display}."
                : $"Panorama refused {display}: {detail}");
        }
    }

    /// <summary>Reads a <c>?method=json</c> listing of <paramref name="folder"/>.</summary>
    /// <remarks>
    /// Each entry's path is built from the parent and its name rather than the server's href, which
    /// arrives encoded in more than one form (as PanoramaBridge's MethodJsonParser found).
    /// </remarks>
    public static IReadOnlyList<PanoramaEntry> ParseListing(string body, string folder)
    {
        Listing? listing;
        try
        {
            listing = JsonSerializer.Deserialize<Listing>(body, ListingOptions);
        }
        catch (JsonException ex)
        {
            // An expired session or a missing sign-in answers with an HTML page.
            throw new PanoramaException("Panorama did not send a folder listing; the sign-in was probably not accepted.", ex)
            {
                IsSignInProblem = true,
            };
        }

        if (listing?.Files is null)
            throw new PanoramaException("Panorama's folder listing was empty or not understood.");

        var parent = PanoramaPaths.ToFolder(folder);
        return listing.Files
            .Where(f => !string.IsNullOrEmpty(f.Text))
            .Select(f =>
            {
                var isFolder = f.Collection ?? !(f.Leaf ?? true);
                return new PanoramaEntry(f.Text!, PanoramaPaths.Combine(parent, f.Text!), isFolder,
                    f.ContentLength ?? f.Size ?? 0, ParseTimestamp(f.LastModified), f.CanUpload ?? false);
            })
            .ToList();
    }

    /// <summary>
    /// Reads <c>?method=md5sum</c>: one <c>hash *name</c> line per file, names verbatim (no
    /// percent-encoding or escaping - measured by PanoramaBridge against panoramaweb.org).
    /// </summary>
    public static IReadOnlyDictionary<string, string> ParseMd5s(string body)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var raw in body.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length < 34 || line[32] != ' ')
                continue;
            var hash = line[..32];
            if (!hash.All(Uri.IsHexDigit))
                continue;
            var name = line[33..];
            if (name.StartsWith('*'))
                name = name[1..];
            result[name] = hash.ToLowerInvariant();
        }

        return result;
    }

    /// <summary>The reason in a LabKey API refusal (<c>{"exception": "..."}</c>), if the body is one.</summary>
    private static string? ApiError(string body)
    {
        try
        {
            using var document = JsonDocument.Parse(body);
            return document.RootElement.ValueKind == JsonValueKind.Object
                   && document.RootElement.TryGetProperty("exception", out var e) && e.ValueKind == JsonValueKind.String
                ? e.GetString()
                : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static DateTimeOffset? ParseTimestamp(string? value) =>
        !string.IsNullOrWhiteSpace(value)
        && DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture,
            DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var parsed)
            ? parsed
            : null;

    public void Dispose() => _http.Dispose();

    private static readonly JsonSerializerOptions ListingOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true,
        NumberHandling = JsonNumberHandling.AllowReadingFromString,
    };

    private sealed class Listing
    {
        [JsonPropertyName("files")]
        public List<Entry>? Files { get; set; }
    }

    private sealed class Entry
    {
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("collection")]
        public bool? Collection { get; set; }

        [JsonPropertyName("leaf")]
        public bool? Leaf { get; set; }

        [JsonPropertyName("contentlength")]
        public long? ContentLength { get; set; }

        [JsonPropertyName("size")]
        public long? Size { get; set; }

        [JsonPropertyName("lastmodified")]
        public string? LastModified { get; set; }

        [JsonPropertyName("canUpload")]
        public bool? CanUpload { get; set; }
    }
}
