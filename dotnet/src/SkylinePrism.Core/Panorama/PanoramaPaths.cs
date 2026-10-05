using System;
using System.Linq;
using System.Text.RegularExpressions;

namespace SkylinePrism.Core.Panorama;

/// <summary>
/// Panorama paths three ways: WebDAV (<c>/_webdav/MacCoss/maccoss/@files/RawFiles/</c>, what a
/// listing and an upload use), folder (<c>/MacCoss/maccoss/@files/RawFiles</c>, what a person
/// writes and what <c>panorama.json</c> records), and the address a browser opens.
/// </summary>
/// <remarks>Ported from LabOps (<c>LabOps.Core.Panorama.PanoramaPaths</c>), MacCoss Lab.</remarks>
public static class PanoramaPaths
{
    public static Uri DefaultServer { get; } = new("https://panoramaweb.org");

    /// <summary>Where browsing starts: the MacCoss project, which holds the lab's shared folder.</summary>
    public const string StartFolder = "/MacCoss";

    private const string WebDavRoot = "/_webdav";

    /// <summary>With a leading and a trailing slash.</summary>
    public static string AsFolder(string path)
    {
        var trimmed = path.Trim().Trim('/');
        return "/" + trimmed + (trimmed.Length == 0 ? "" : "/");
    }

    /// <summary><c>/_webdav/MacCoss/maccoss/@files/X/</c> to <c>/MacCoss/maccoss/@files/X</c>.</summary>
    public static string ToFolder(string webDavPath)
    {
        var path = "/" + webDavPath.Trim().Trim('/');
        return path.StartsWith(WebDavRoot + "/", StringComparison.OrdinalIgnoreCase)
            ? path[WebDavRoot.Length..]
            : path == WebDavRoot ? "/" : path;
    }

    /// <summary><c>/MacCoss/maccoss/@files/X</c> to <c>/_webdav/MacCoss/maccoss/@files/X/</c>.</summary>
    public static string ToWebDav(string folder) => AsFolder(WebDavRoot + "/" + ToFolder(folder).Trim('/'));

    /// <summary>
    /// Each segment percent-encoded, except '@', which LabKey's file roots (<c>@files</c>) keep as is
    /// (PanoramaBridge's RemotePath does the same).
    /// </summary>
    public static string Encode(string path) =>
        string.Join('/', path.Split('/').Select(s => Uri.EscapeDataString(s).Replace("%40", "@", StringComparison.Ordinal)));

    /// <summary>The container part of a folder: everything before its file root (<c>@files</c>).</summary>
    public static string Container(string folder)
    {
        var segments = ToFolder(folder).Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        var root = Array.FindIndex(segments, s => s.StartsWith('@'));
        return "/" + string.Join('/', root < 0 ? segments : segments[..root]);
    }

    /// <summary>Whether the folder is inside a file root (<c>@files</c>), where files can be uploaded.</summary>
    public static bool IsFileArea(string folder) =>
        ToFolder(folder).Split('/').Any(s => s.StartsWith('@'));

    /// <summary>The folder one level up; the root's parent is the root.</summary>
    public static string Parent(string folder)
    {
        var segments = ToFolder(folder).Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        return "/" + string.Join('/', segments.Take(Math.Max(0, segments.Length - 1)));
    }

    /// <summary>The last segment of a folder.</summary>
    public static string Name(string folder) =>
        ToFolder(folder).Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries).LastOrDefault() ?? "";

    /// <summary>A folder joined with a child name.</summary>
    public static string Combine(string folder, string child) =>
        "/" + (ToFolder(folder).Trim('/') + "/" + child.Trim('/')).Trim('/');

    /// <summary>
    /// A folder from whatever a person pasted: a folder path, a WebDAV path, or a browser address
    /// such as <c>https://panoramaweb.org/MacCoss/maccoss/X/project-begin.view</c> - the page the
    /// address names (<c>*.view</c>, <c>*.api</c>) and any query are dropped.
    /// </summary>
    public static string FromAnything(string text)
    {
        var s = text.Trim();
        if (Uri.TryCreate(s, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https")
            s = Uri.UnescapeDataString(uri.AbsolutePath);
        s = Regex.Replace(s, @"/[^/]+\.(view|api|post)$", "", RegexOptions.IgnoreCase);
        return ToFolder(s);
    }

    /// <summary>The page a browser opens for a folder: a file area as its WebDAV listing, a container as its start page.</summary>
    public static string BrowserUrl(string folder, Uri? server = null)
    {
        var host = (server ?? DefaultServer).GetLeftPart(UriPartial.Authority);
        var trimmed = ToFolder(folder).Trim('/');
        return IsFileArea(trimmed)
            ? host + Encode(ToWebDav(trimmed))
            : $"{host}/{Encode(trimmed)}/project-begin.view";
    }

    /// <summary>Where a wiki page opens.</summary>
    public static string WikiPageUrl(string folder, string pageName, Uri? server = null) =>
        (server ?? DefaultServer).GetLeftPart(UriPartial.Authority)
        + Encode(AsFolder(Container(folder))) + "wiki-page.view?name=" + Uri.EscapeDataString(pageName);
}
