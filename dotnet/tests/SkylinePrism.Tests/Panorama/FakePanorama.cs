using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace SkylinePrism.Tests.Panorama;

/// <summary>
/// An in-memory Panorama (LabKey) with the behaviors PRISM depends on, each as it was measured on
/// panoramaweb.org rather than as a specification says: MKCOL single-level (409 for a missing parent,
/// 200 for a folder already there), PUT answering 201 for a replacement too, an <c>.html</c> upload
/// refused (403) when it carries a doctype or a style/link/script element, <c>?method=md5sum</c> over
/// the stored bytes, the wiki editor's <c>LABKEY._wiki.setProps</c>, a save refused without a CSRF
/// token or against a stale page version, and an attachment refused with a warning when its name is
/// already taken. And the folder page's web parts: <c>project-addWebPart.view</c> a form action that
/// redirects, taking the column by its INTERNAL name (<c>!content</c> for the body) and saving a part
/// under any other name too, where no page renders it and <c>project-getWebParts.api</c> leaves it out.
/// </summary>
internal sealed class FakePanorama : HttpMessageHandler
{
    public const string Csrf = "csrf-token-1";

    /// <summary>A web part row, with its column under LabKey's internal name.</summary>
    public sealed class WebPartRow
    {
        public required int Id { get; init; }
        public required string Container { get; init; }
        public required string PageId { get; init; }
        public required string Location { get; init; }
        public required string Name { get; init; }
        public int Index { get; set; }
        public Dictionary<string, string> Properties { get; } = new(StringComparer.Ordinal);
    }

    public List<WebPartRow> WebParts { get; } = new();

    /// <summary>Folder types by container; a container not listed is an ordinary ("Collaboration") folder.</summary>
    public Dictionary<string, string> FolderTypes { get; } = new(StringComparer.Ordinal);

    /// <summary>Whether the signed-in user administers the folders (adding a web part needs it).</summary>
    public bool IsAdmin { get; set; } = true;

    /// <summary>Makes customizing a web part fail, as a dropped connection or a server error would.</summary>
    public bool FailCustomize { get; set; }

    /// <summary>Files attached to wiki pages, over the fake's life.</summary>
    public int FilesAttached { get; private set; }

    /// <summary>Attachments a saved page body pointed at that the page did not have when it was saved.</summary>
    public List<string> MissingAtSave { get; } = new();

    private int _nextWebPartId = 1000;

    /// <summary>The page a browser opens for the folder: the dashboard of a Panorama folder, the portal page otherwise.</summary>
    public string MainPage(string container) =>
        FolderTypes.GetValueOrDefault(container) == "Targeted MS" ? "DefaultDashboard" : "portal.default";

    public WebPartRow AddWebPart(string container, string pageId, string location, string name)
    {
        var row = new WebPartRow
        {
            Id = _nextWebPartId++, Container = container, PageId = pageId, Location = location, Name = name,
            // After every part already on the page, as Portal.addPart does (it renumbers on save, so
            // its "parts.size()" is the last place; here a removed part can leave a gap).
            Index = WebParts.Where(p => p.Container == container && p.PageId == pageId).Select(p => p.Index).DefaultIfEmpty(0).Max() + 1,
        };
        WebParts.Add(row);
        return row;
    }

    /// <summary>The parts a browser sees on the folder's page, top to bottom, body column first.</summary>
    public IReadOnlyList<WebPartRow> RenderedParts(string container) =>
        WebParts.Where(p => p.Container == container && p.PageId == MainPage(container) && Friendly(p.Location) is not null)
            .OrderBy(p => p.Location == "!content" ? 0 : 1).ThenBy(p => p.Index).ToList();

    // SimpleWebPartFactory.getFriendlyLocationName: anything else is dropped from the listing.
    private static string? Friendly(string location) => location switch
    {
        "!content" => "body",
        "right" => "right",
        "menubar" => "menu",
        _ => null,
    };

    public sealed class Page
    {
        public required string EntityId { get; init; }
        public required string Name { get; init; }
        public string Title { get; set; } = "";
        public string Body { get; set; } = "";
        public int Version { get; set; } = 1;
        public bool ShowAttachments { get; set; } = true;
        public Dictionary<string, byte[]> Attachments { get; } = new(StringComparer.Ordinal);
    }

    /// <summary>Files by full folder path (<c>/P/@files/x/name</c>).</summary>
    public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);

    /// <summary>Folders that exist; containers and their <c>@files</c> roots are seeded by the test.</summary>
    public HashSet<string> Folders { get; } = new(StringComparer.Ordinal) { "/" };

    /// <summary>Wiki pages by container + "|" + name.</summary>
    public Dictionary<string, Page> Pages { get; } = new(StringComparer.Ordinal);

    public List<string> Requests { get; } = new();

    public bool RejectSignIn { get; set; }

    /// <summary>Rewrites what a PUT stores, to simulate a transfer that arrives damaged.</summary>
    public Func<string, byte[], byte[]>? DamageOnPut { get; set; }

    public void AddContainer(string container)
    {
        var parts = container.Trim('/').Split('/');
        for (var i = 1; i <= parts.Length; i++)
            Folders.Add("/" + string.Join('/', parts.Take(i)));
        Folders.Add(container.TrimEnd('/') + "/@files");
    }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var uri = request.RequestUri!;
        var path = Uri.UnescapeDataString(uri.AbsolutePath);
        var query = uri.Query.TrimStart('?');
        Requests.Add($"{request.Method} {path}{(query.Length > 0 ? "?" + query : "")}");

        if (RejectSignIn)
            return new HttpResponseMessage(HttpStatusCode.Unauthorized);

        if (path == "/_webdav" || path.StartsWith("/_webdav/", StringComparison.Ordinal))
            return await WebDavAsync(request, "/" + path["/_webdav".Length..].Trim('/'), query, cancellationToken);

        var slash = path.LastIndexOf('/');
        var container = path[..slash];
        var action = path[(slash + 1)..];
        return action switch
        {
            "login-whoami.api" => WhoAmI(),
            "wiki-edit.view" => WikiEditor(container, Query(query, "name")),
            "wiki-saveWiki.api" => await SaveWikiAsync(request, container, cancellationToken),
            "wiki-attachFiles.api" => await AttachAsync(request, container, cancellationToken),
            "project-begin.view" => Begin(container),
            "project-getContainers.api" => Json(new { id = "id-of" + container, path = container, folderType = FolderTypes.GetValueOrDefault(container, "Collaboration") }),
            "project-getWebParts.api" => Layout(container, Query(query, "pageId")),
            "project-addWebPart.view" => await AddWebPartAsync(request, container, cancellationToken),
            "project-customizeWebPartAsync.api" => await CustomizeAsync(request, container, cancellationToken),
            "project-moveWebPartAsync.api" => await MoveAsync(request, container, cancellationToken),
            _ => new HttpResponseMessage(HttpStatusCode.NotFound),
        };
    }

    private HttpResponseMessage Begin(string container)
    {
        var parts = string.Concat(RenderedParts(container).Select(p => $"<div id=\"webpart_{p.Id}\" class=\"panel\"></div>"));
        var html = $"<html><head><script>LABKEY.container = {{\"folderType\":\"{FolderTypes.GetValueOrDefault(container, "Collaboration")}\"}};</script></head>"
                   + $"<body>{parts}</body></html>";
        return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(html) };
    }

    private HttpResponseMessage Layout(string container, string pageId)
    {
        var layout = WebParts.Where(p => p.Container == container && p.PageId == pageId && Friendly(p.Location) is not null)
            .OrderBy(p => p.Index)
            .GroupBy(p => Friendly(p.Location)!)
            .ToDictionary(g => g.Key, g => g.Select(p => new { name = p.Name, index = p.Index, webPartId = p.Id }).ToList());
        return Json(layout);
    }

    private async Task<HttpResponseMessage> AddWebPartAsync(HttpRequestMessage request, string container, CancellationToken ct)
    {
        if (!HasCsrf(request))
            return new HttpResponseMessage(HttpStatusCode.Unauthorized);
        if (!IsAdmin)
            return new HttpResponseMessage(HttpStatusCode.Forbidden);
        var form = await FormAsync(request, ct);
        // Saved whatever the column is called, as LabKey's Portal.addPart does.
        AddWebPart(container, form["pageId"], form["location"], form["name"]);
        var response = new HttpResponseMessage(HttpStatusCode.Found);
        response.Headers.Location = new Uri(form.GetValueOrDefault("returnUrl", container + "/project-begin.view"), UriKind.Relative);
        return response;
    }

    private async Task<HttpResponseMessage> CustomizeAsync(HttpRequestMessage request, string container, CancellationToken ct)
    {
        if (!HasCsrf(request))
            return Json(new { exception = "Invalid security context" }, HttpStatusCode.Unauthorized);
        if (!IsAdmin)
            return Json(new { exception = "User does not have permission to perform this operation." }, HttpStatusCode.Forbidden);
        if (FailCustomize)
            return Json(new { exception = "Something went wrong." }, HttpStatusCode.InternalServerError);
        var form = await FormAsync(request, ct);
        var part = WebParts.FirstOrDefault(p => p.Container == container && p.Id.ToString(CultureInfo.InvariantCulture) == form["webPartId"]);
        if (part is null)
            return Json(new { success = false });
        // CustomizeWebPartHelper.populatePropertyMap: the whole map is replaced.
        part.Properties.Clear();
        foreach (var (key, value) in form.Where(f => f.Key is not ("webPartId" or "index" or "pageId" or "x" or "y" or "returnUrl")))
            part.Properties[key] = value;
        return Json(new { success = true });
    }

    private async Task<HttpResponseMessage> MoveAsync(HttpRequestMessage request, string container, CancellationToken ct)
    {
        if (!HasCsrf(request))
            return Json(new { exception = "Invalid security context" }, HttpStatusCode.Unauthorized);
        var form = await FormAsync(request, ct);
        var part = WebParts.FirstOrDefault(p => p.Container == container && p.Id.ToString(CultureInfo.InvariantCulture) == form["webPartId"]);
        if (part is null)
            return Json(new { exception = "Unable to move the specified web part." }, HttpStatusCode.NotFound);
        // ProjectController.handleMoveWebPart: swaps with the nearest part above it in the same column.
        var up = form["direction"] == "0";
        var neighbor = WebParts.Where(p => p.Container == container && p.PageId == part.PageId && p.Location == part.Location
                                           && (up ? p.Index < part.Index : p.Index > part.Index))
            .OrderBy(p => up ? -p.Index : p.Index).FirstOrDefault();
        if (neighbor is not null)
            (part.Index, neighbor.Index) = (neighbor.Index, part.Index);
        return Layout(container, part.PageId);
    }

    private static async Task<Dictionary<string, string>> FormAsync(HttpRequestMessage request, CancellationToken ct) =>
        (await request.Content!.ReadAsStringAsync(ct)).Split('&', StringSplitOptions.RemoveEmptyEntries)
        .Select(p => p.Split('=', 2))
        .ToDictionary(p => WebUtility.UrlDecode(p[0]), p => WebUtility.UrlDecode(p.Length > 1 ? p[1] : ""), StringComparer.Ordinal);

    private async Task<HttpResponseMessage> WebDavAsync(HttpRequestMessage request, string folder, string query, CancellationToken ct)
    {
        folder = folder == "" ? "/" : folder;
        if (request.Method == HttpMethod.Get && query == "method=json")
        {
            if (!Folders.Contains(folder))
                return new HttpResponseMessage(HttpStatusCode.NotFound);
            var prefix = folder == "/" ? "/" : folder + "/";
            var entries = Folders.Where(f => f != folder && f.StartsWith(prefix, StringComparison.Ordinal) && !f[prefix.Length..].Contains('/'))
                .Select(f => (object)new { text = f[prefix.Length..], collection = true, canUpload = true })
                .Concat(Files.Where(f => f.Key.StartsWith(prefix, StringComparison.Ordinal) && !f.Key[prefix.Length..].Contains('/'))
                    .Select(f => (object)new { text = f.Key[prefix.Length..], collection = false, size = f.Value.Length, canUpload = true }));
            return Json(new { files = entries });
        }

        if (request.Method == HttpMethod.Get && query == "method=md5sum")
        {
            if (!Folders.Contains(folder))
                return new HttpResponseMessage(HttpStatusCode.NotFound);
            var lines = Files.Where(f => Parent(f.Key) == folder)
                .Select(f => $"{Convert.ToHexStringLower(MD5.HashData(f.Value))} *{f.Key[(folder.Length + 1)..]}");
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(string.Join("\n", lines) + "\n") };
        }

        if (request.Method.Method == "MKCOL")
        {
            if (Folders.Contains(folder))
                return new HttpResponseMessage(HttpStatusCode.OK); // measured: 200, not 405
            if (!Folders.Contains(Parent(folder)))
                return new HttpResponseMessage(HttpStatusCode.Conflict);
            Folders.Add(folder);
            return new HttpResponseMessage(HttpStatusCode.Created);
        }

        if (request.Method == HttpMethod.Put)
        {
            if (!Folders.Contains(Parent(folder)))
                return new HttpResponseMessage(HttpStatusCode.Conflict);
            var bytes = await request.Content!.ReadAsByteArrayAsync(ct);
            if (folder.EndsWith(".html", StringComparison.OrdinalIgnoreCase)
                && Regex.IsMatch(Encoding.UTF8.GetString(bytes), @"<!doctype|<style|<link|<script|<iframe|<form|\son[a-z]+\s*=", RegexOptions.IgnoreCase))
                return new HttpResponseMessage(HttpStatusCode.Forbidden);
            Files[folder] = DamageOnPut?.Invoke(folder, bytes) ?? bytes;
            return new HttpResponseMessage(HttpStatusCode.Created);
        }

        return new HttpResponseMessage(HttpStatusCode.MethodNotAllowed);
    }

    private static HttpResponseMessage WhoAmI()
    {
        var response = Json(new { CSRF = Csrf, displayName = "tester" });
        response.Headers.Add("Set-Cookie", "JSESSIONID=session-1; Path=/; HttpOnly");
        return response;
    }

    private HttpResponseMessage WikiEditor(string container, string name)
    {
        if (!Pages.TryGetValue(container + "|" + name, out var page))
            return new HttpResponseMessage(HttpStatusCode.NotFound);
        var html = "<html><body><script type=\"text/javascript\">\nLABKEY._wiki.setProps({\n"
                   + $"    entityId: '{page.EntityId}',\n    rowId: 7,\n    pageVersionId: {page.Version},\n"
                   + $"    name: '{Js(page.Name)}',\n    title: '{Js(page.Title)}',\n    body: '{Js(page.Body)}',\n"
                   + $"    parent: -1,\n    showAttachments: {(page.ShowAttachments ? "true" : "false")},\n    shouldIndex: true\n"
                   + "});\n</script></body></html>";
        return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(html) };
    }

    private async Task<HttpResponseMessage> SaveWikiAsync(HttpRequestMessage request, string container, CancellationToken ct)
    {
        if (!HasCsrf(request))
            return Json(new { exception = "Invalid security context" }, HttpStatusCode.Forbidden);
        using var json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(ct));
        var root = json.RootElement;
        var name = root.GetProperty("name").GetString()!;
        var key = container + "|" + name;
        if (!Pages.TryGetValue(key, out var page))
            Pages[key] = page = new Page { EntityId = Guid.NewGuid().ToString(), Name = name };
        else if (root.TryGetProperty("pageVersionId", out var v) && v.ValueKind == JsonValueKind.Number && v.GetInt32() != page.Version)
            return Json(new { exception = "Someone else has edited this page." }, HttpStatusCode.BadRequest);
        else
            page.Version++;
        page.Title = root.GetProperty("title").GetString() ?? "";
        page.Body = root.GetProperty("body").GetString() ?? "";
        foreach (Match m in Regex.Matches(page.Body, @"wiki-download\.view\?entityId=[^&""]*&(?:amp;)?name=([^""&]+)"))
            if (!page.Attachments.ContainsKey(Uri.UnescapeDataString(m.Groups[1].Value)))
                MissingAtSave.Add(Uri.UnescapeDataString(m.Groups[1].Value));
        page.ShowAttachments = root.GetProperty("showAttachments").GetBoolean();
        return Json(new { success = true, wikiProps = new { entityId = page.EntityId, rowId = 7, pageVersionId = page.Version } });
    }

    private async Task<HttpResponseMessage> AttachAsync(HttpRequestMessage request, string container, CancellationToken ct)
    {
        if (!HasCsrf(request))
            return Json(new { exception = "Invalid security context" }, HttpStatusCode.Forbidden);
        if (request.Content is not MultipartFormDataContent form)
            return Json(new { exception = "You must use the 'multipart/form-data' mimetype when posting to attachFiles.api" },
                HttpStatusCode.BadRequest);

        string? entityId = null;
        var delete = new List<string>();
        var add = new List<(string Name, byte[] Data)>();
        foreach (var part in form)
        {
            var disposition = part.Headers.ContentDisposition!;
            var field = disposition.Name!.Trim('"');
            if (field == "entityId")
                entityId = await part.ReadAsStringAsync(ct);
            else if (field == "toDelete")
                delete.Add(await part.ReadAsStringAsync(ct));
            else
                add.Add((disposition.FileName!.Trim('"'), await part.ReadAsByteArrayAsync(ct)));
        }

        var page = Pages.Values.FirstOrDefault(p => p.EntityId == entityId);
        if (page is null)
            return Json(new { exception = "No wiki page with that entity id." }, HttpStatusCode.NotFound);
        foreach (var name in delete)
            page.Attachments.Remove(name);
        var warnings = new List<string>();
        foreach (var (name, data) in add)
        {
            if (page.Attachments.ContainsKey(name))
                warnings.Add($"A file with name {name} already exists.");
            else
            {
                page.Attachments[name] = data;
                FilesAttached++;
            }
        }

        return Json(new
        {
            success = true,
            warnings = warnings.Count == 0 ? null : new { files = string.Join(" ", warnings) },
            attachments = page.Attachments.Keys.Select(n => new
            {
                name = n,
                downloadUrl = $"https://panorama.test{container}/wiki-download.view?entityId={page.EntityId}&name={Uri.EscapeDataString(n)}",
            }),
        });
    }

    private static bool HasCsrf(HttpRequestMessage request) =>
        request.Headers.TryGetValues("X-LABKEY-CSRF", out var token) && token.Single() == Csrf
        && request.Headers.TryGetValues("Cookie", out var cookie) && cookie.Single().Contains("JSESSIONID=session-1", StringComparison.Ordinal);

    private static string Parent(string path) => path.LastIndexOf('/') is var i and > 0 ? path[..i] : "/";

    private static string Query(string query, string key) =>
        query.Split('&').Select(p => p.Split('=', 2)).Where(p => p[0] == key).Select(p => Uri.UnescapeDataString(p[1])).FirstOrDefault() ?? "";

    /// <summary>A JavaScript single-quoted literal, the way LabKey writes the editor's props.</summary>
    private static string Js(string text) =>
        text.Replace("\\", "\\\\").Replace("'", "\\'").Replace("\n", "\\n").Replace("\r", "\\r").Replace("<", "\\x3C");

    private static HttpResponseMessage Json(object value, HttpStatusCode status = HttpStatusCode.OK) =>
        new(status)
        {
            Content = new StringContent(JsonSerializer.Serialize(value, new JsonSerializerOptions
            {
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
            }), Encoding.UTF8, "application/json"),
        };
}
