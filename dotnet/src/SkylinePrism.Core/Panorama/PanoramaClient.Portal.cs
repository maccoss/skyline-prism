using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace SkylinePrism.Core.Panorama;

/// <summary>A web part on a folder's page, as <c>project-getWebParts.api</c> lists it.</summary>
public sealed record WebPart(string Location, string Name, int Index, int WebPartId);

public sealed partial class PanoramaClient
{
    /// <summary>The page id of an ordinary LabKey folder's main page (<c>Portal.DEFAULT_PORTAL_PAGE_ID</c>).</summary>
    public const string DefaultPortalPage = "portal.default";

    /// <summary>
    /// The page id of a Panorama (Targeted MS) folder's dashboard: the targetedms module saves its parts
    /// under <c>DefaultFolderType.DEFAULT_DASHBOARD</c>, and its tabs are named pages beside it
    /// ("Runs", "Raw Data", "Data Pipeline"). Measured on panoramaweb.org: <c>portal.default</c>
    /// lists nothing in such a folder.
    /// </summary>
    public const string PanoramaDashboardPage = "DefaultDashboard";

    /// <summary>The container's id, which a Wiki web part names its page's folder by.</summary>
    public async Task<string> ContainerIdAsync(string container, CancellationToken cancellationToken = default)
    {
        var body = await GetAsync(PortalUrl(container, "project-getContainers.api?includeSubfolders=false"),
            PanoramaPaths.Container(container), cancellationToken).ConfigureAwait(false);
        using var document = ParseJson(body!, "the folder's details");
        return Text(document.RootElement, "id")
               ?? throw new PanoramaException($"Panorama did not give the id of {PanoramaPaths.Container(container)}.");
    }

    /// <summary>The web parts on one page of a folder.</summary>
    public async Task<IReadOnlyList<WebPart>> WebPartsAsync(string container, string pageId, CancellationToken cancellationToken = default)
    {
        var body = await GetAsync(PortalUrl(container, "project-getWebParts.api?pageId=" + Uri.EscapeDataString(pageId)),
            $"the web parts of {PanoramaPaths.Container(container)}", cancellationToken).ConfigureAwait(false);
        return ParseWebParts(body!);
    }

    /// <summary>
    /// The id of the page a browser opens for the folder: the candidate whose web parts are the ones the
    /// folder's start page actually renders. A folder with nothing on its page falls back to its folder
    /// type - the Panorama dashboard for a Targeted MS folder, the portal page for anything else.
    /// </summary>
    public async Task<string> MainPageIdAsync(string container, CancellationToken cancellationToken = default)
    {
        var start = await GetAsync(PortalUrl(container, "project-begin.view"), PanoramaPaths.Container(container), cancellationToken)
            .ConfigureAwait(false);
        var rendered = Regex.Matches(start!, @"id=""webpart_(\d+)""").Select(m => int.Parse(m.Groups[1].Value)).ToHashSet();
        foreach (var candidate in new[] { PanoramaDashboardPage, DefaultPortalPage })
        {
            var parts = await WebPartsAsync(container, candidate, cancellationToken).ConfigureAwait(false);
            if (parts.Any(p => rendered.Contains(p.WebPartId)))
                return candidate;
        }

        return start!.Contains("\"folderType\":\"Targeted MS\"", StringComparison.Ordinal) || start.Contains("PanoramaDashboardTab", StringComparison.Ordinal)
            ? PanoramaDashboardPage
            : DefaultPortalPage;
    }

    /// <summary>
    /// Adds a web part to a page and returns its id. Needs folder administrator permission.
    /// </summary>
    /// <param name="location">The column, as <see cref="WebPartsAsync"/> names it: <c>body</c> or <c>right</c>.</param>
    /// <remarks>
    /// <para>
    /// <c>project-addWebPart.view</c> is a form action, not an API: it answers with a redirect and never
    /// names the new part's id, so the id is found by listing the page before and after.
    /// </para>
    /// <para>
    /// It also takes the column by LabKey's INTERNAL name, which for the body is <c>!content</c>, while
    /// the listing reports it as <c>body</c> (<c>SimpleWebPartFactory.getFriendlyLocationName</c>).
    /// Sent <c>body</c>, the action saves the part anyway, in a column no page renders and the listing
    /// leaves out - so it answers exactly as a success does and nothing appears. Measured on
    /// panoramaweb.org; the internal name is the one LabKey's own Add Web Part form posts.
    /// </para>
    /// </remarks>
    public async Task<int> AddWebPartAsync(string container, string pageId, string name, string location,
        CancellationToken cancellationToken = default)
    {
        var internalLocation = InternalLocation(location);
        var before = (await WebPartsAsync(container, pageId, cancellationToken).ConfigureAwait(false)).Select(p => p.WebPartId).ToHashSet();
        await PostFormAsync(PortalUrl(container, "project-addWebPart.view"), new Dictionary<string, string>
        {
            ["pageId"] = pageId,
            ["location"] = internalLocation,
            ["name"] = name,
            ["returnUrl"] = PanoramaPaths.Encode(PanoramaPaths.AsFolder(PanoramaPaths.Container(container))) + "project-begin.view",
        }, $"a {name} web part on {PanoramaPaths.Container(container)}'s page", cancellationToken).ConfigureAwait(false);

        var added = (await WebPartsAsync(container, pageId, cancellationToken).ConfigureAwait(false))
            .Where(p => !before.Contains(p.WebPartId) && p.Name == name)
            .ToList();
        return added.Count == 1
            ? added[0].WebPartId
            : throw new PanoramaException($"Panorama did not add the {name} web part to {PanoramaPaths.Container(container)}'s page.");
    }

    /// <summary>LabKey's internal name for a page column (<c>WebPartFactory.LOCATION_BODY</c> is <c>!content</c>).</summary>
    internal static string InternalLocation(string location) => location switch
    {
        "body" => "!content",
        "right" => "right",
        "menu" => "menubar",
        _ => throw new ArgumentException($"'{location}' is not a page column (body, right or menu).", nameof(location)),
    };

    /// <summary>
    /// Sets a web part's properties (for a Wiki part: <c>name</c>, the page, and <c>webPartContainer</c>,
    /// the id of the folder holding it), replacing the whole property map, as LabKey's
    /// <c>populatePropertyMap</c> does.
    /// </summary>
    public async Task CustomizeWebPartAsync(string container, int webPartId, IReadOnlyDictionary<string, string> properties,
        CancellationToken cancellationToken = default)
    {
        var display = $"the web part on {PanoramaPaths.Container(container)}'s page";
        var form = new Dictionary<string, string>(properties) { ["webPartId"] = webPartId.ToString(System.Globalization.CultureInfo.InvariantCulture) };
        var body = await PostWithCsrfAsync(PortalUrl(container, "project-customizeWebPartAsync.api"),
            () => new FormUrlEncodedContent(form), display, cancellationToken).ConfigureAwait(false);
        using var document = ParseJson(body, display);
        // A part removed in the meantime is not an HTTP error: the answer says success false.
        if (!(document.RootElement.TryGetProperty("success", out var success) && success.ValueKind == JsonValueKind.True))
            throw new PanoramaException($"Panorama did not change {display}: the web part may have been removed.");
    }

    /// <summary>Moves a web part one place up within its column (<c>Portal.MOVE_UP</c> = 0).</summary>
    public async Task MoveWebPartUpAsync(string container, int webPartId, CancellationToken cancellationToken = default) =>
        await PostWithCsrfAsync(PortalUrl(container, "project-moveWebPartAsync.api"),
            () => new FormUrlEncodedContent(new Dictionary<string, string>
            {
                ["webPartId"] = webPartId.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["direction"] = "0",
            }),
            $"the web part on {PanoramaPaths.Container(container)}'s page", cancellationToken).ConfigureAwait(false);

    /// <summary>
    /// A POST to a LabKey form action, which answers success with a redirect. Unlike an API call, a 3xx
    /// here is the expected answer - except one to the sign-in page, which is still a refusal.
    /// </summary>
    private async Task PostFormAsync(string url, IReadOnlyDictionary<string, string> form, string display,
        CancellationToken cancellationToken)
    {
        var (token, cookie) = await SecurityContextAsync(cancellationToken).ConfigureAwait(false);
        using var request = new HttpRequestMessage(HttpMethod.Post, url) { Content = new FormUrlEncodedContent(form) };
        if (token is not null)
            request.Headers.TryAddWithoutValidation("X-LABKEY-CSRF", token);
        if (cookie is not null)
            request.Headers.TryAddWithoutValidation("Cookie", cookie);
        using var response = await SendRawAsync(request, display, MetadataTimeout, cancellationToken).ConfigureAwait(false);
        var redirect = (int)response.StatusCode is >= 300 and < 400;
        var toSignIn = response.Headers.Location?.OriginalString.Contains("login", StringComparison.OrdinalIgnoreCase) == true;
        if (redirect && !toSignIn)
            return;
        if (response.StatusCode == HttpStatusCode.OK)
            // A form action that does not redirect has redrawn its form with an error on it.
            throw new PanoramaException($"Panorama did not accept the change to {display}.");
        await EnsureSuccessAsync(response, display, cancellationToken).ConfigureAwait(false);
    }

    private static string PortalUrl(string container, string action) =>
        PanoramaPaths.Encode(PanoramaPaths.AsFolder(PanoramaPaths.Container(container))) + action;

    /// <summary>Reads <c>project-getWebParts.api</c>: <c>{"body": [{name, index, webPartId}], "right": [...]}</c>.</summary>
    public static IReadOnlyList<WebPart> ParseWebParts(string body)
    {
        using var document = ParseJson(body, "the folder's web parts");
        var parts = new List<WebPart>();
        foreach (var location in document.RootElement.EnumerateObject().Where(l => l.Value.ValueKind == JsonValueKind.Array))
            foreach (var part in location.Value.EnumerateArray())
                if (Int(part, "webPartId") is { } id)
                    parts.Add(new WebPart(location.Name, Text(part, "name") ?? "", Int(part, "index") ?? 0, id));
        return parts.OrderBy(p => p.Index).ToList();
    }
}
