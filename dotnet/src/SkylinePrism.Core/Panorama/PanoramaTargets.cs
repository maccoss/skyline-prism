using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SkylinePrism.Core.Panorama;

/// <summary>Where one report's wiki page goes: a folder (its container holds the page) and a page name.</summary>
public sealed record WikiTarget(string Folder, string Page);

/// <summary>A quant page published from this output directory, and the contrast it shows.</summary>
public sealed record QuantPage(string Folder, string Page, string? Contrast);

/// <summary>
/// Where an output directory was last published, kept beside the outputs as <see cref="FileName"/>
/// so the CLI and the GUI remember it for each output directory and agree on it.
/// </summary>
/// <remarks>
/// <para>Holds folders and page names only - never a credential. The file lives in the output
/// directory, which is itself uploaded, and a key written here would go to Panorama with it.</para>
/// <para>It is a convenience, not provenance: nothing reads it to decide what a result means, and an
/// output directory without one publishes exactly the same once its targets are given.</para>
/// </remarks>
/// <param name="LinksWiki">The links page, which the folder's own page shows in a Wiki web part.</param>
/// <param name="LinksWebPartId">The web part PRISM added to show it, so a republish updates that part rather than adding another.</param>
/// <param name="QuantPages">Every quant page published from this output directory - one per contrast - for the links page.</param>
/// <param name="PublishId">
/// This output directory's id, written into the footer of every page it publishes, so a different
/// output directory with the same name - and so the same default page names - cannot replace its
/// pages unnoticed. Derived from the run on its first publish (<see cref="OutputPublishing.DerivedPublishId"/>)
/// and kept here after. Not secret, and not provenance.
/// </param>
/// <param name="LinksWebPartFolder">The folder whose page <paramref name="LinksWebPartId"/> is on; older files
/// leave it out, and then it is the links page's folder.</param>
/// <param name="LinksWebPartPage">The links page that part shows.</param>
public sealed record PanoramaTargets(
    string Server,
    WikiTarget? QcWiki,
    WikiTarget? QuantWiki,
    string? RawFolder,
    string? Destination,
    WikiTarget? LinksWiki = null,
    int? LinksWebPartId = null,
    IReadOnlyList<QuantPage>? QuantPages = null,
    string? PublishId = null,
    string? LinksWebPartFolder = null,
    string? LinksWebPartPage = null)
{
    public const string FileName = "panorama.json";

    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static PanoramaTargets Empty { get; } = new(PanoramaPaths.DefaultServer.AbsoluteUri.TrimEnd('/'), null, null, null, null);

    /// <summary>
    /// The targets recorded in an output directory, or <see cref="Empty"/> when there are none (or they
    /// cannot be read). For showing what was remembered; a publish uses <see cref="LoadForPublish"/>.
    /// </summary>
    public static PanoramaTargets Load(string outputDir)
    {
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return Empty;
        try
        {
            return JsonSerializer.Deserialize<PanoramaTargets>(ReadShared(path), Json) ?? Empty;
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            // A damaged file is a forgotten preference, not a failure to open a window.
            return Empty;
        }
    }

    /// <summary>The pause before re-reading a file another program holds; the nth retry waits n times this.</summary>
    internal static TimeSpan ReadRetryDelay { get; set; } = TimeSpan.FromMilliseconds(250);

    /// <summary>
    /// The targets as a PUBLISH needs them: like <see cref="Load"/>, except that a file that is there
    /// but cannot be read is retried and then refused, never read as "nothing remembered".
    /// </summary>
    /// <remarks>
    /// <see cref="Load"/>'s leniency is right for a window opening on remembered folders and wrong
    /// here, because the file holds the directory's <see cref="PublishId"/>. A publish that read
    /// nothing through a moment's lock - a scanner, the window reading it, an SMB hiccup - would make
    /// an id and save it over the real one, and from then on every page this directory had published
    /// would be refused as another directory's. A file that is not JSON cannot be read later either,
    /// so it is copied aside (<c>panorama.json.damaged</c>) before anything overwrites it, the publish
    /// goes on as a first one, and says so.
    /// </remarks>
    public static PanoramaTargets LoadForPublish(string outputDir, Action<string>? say = null)
    {
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return Empty;

        string text;
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                text = ReadShared(path);
                break;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                if (attempt < 3)
                {
                    System.Threading.Thread.Sleep(ReadRetryDelay * attempt);
                    continue;
                }

                throw new PanoramaException(
                    $"{FileName} in {outputDir} is there but could not be read ({ex.Message}). Not publishing: it "
                    + "holds the id that says which pages on Panorama are this directory's, and publishing without it "
                    + "would claim them under another. Close whatever has it open and publish again.");
            }
        }

        try
        {
            return JsonSerializer.Deserialize<PanoramaTargets>(text, Json) ?? Empty;
        }
        catch (JsonException ex)
        {
            var aside = path + ".damaged";
            try
            {
                File.Copy(path, aside, overwrite: true);
            }
            catch (Exception copy) when (copy is IOException or UnauthorizedAccessException)
            {
                // Nowhere to put it; the message below still says the file was not used.
            }

            say?.Invoke($"{FileName} in {outputDir} is damaged ({ex.Message}); it was copied to {Path.GetFileName(aside)} "
                        + "and this publish goes on as this directory's first. Pages it published before carry its "
                        + "old id: if they are refused as another directory's, take them over on purpose (--take-over).");
            return Empty;
        }
    }

    // Shared for writing and deletion, so reading the file never holds up the program that owns it
    // (see "A reader must never take a file hostage" in CLAUDE.md).
    private static string ReadShared(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    public void Save(string outputDir) =>
        File.WriteAllText(Path.Combine(outputDir, FileName), JsonSerializer.Serialize(this, Json));

    /// <summary>The links page to publish to: the remembered one, else the default name.</summary>
    public string LinksPageFor(string outputDir) => LinksWiki?.Page ?? PanoramaPublisher.DefaultLinksPage(outputDir);

    /// <summary>Records value equality on the list too, so two loads of one file compare equal.</summary>
    public bool Equals(PanoramaTargets? other) =>
        other is not null && Server == other.Server && QcWiki == other.QcWiki && QuantWiki == other.QuantWiki
        && RawFolder == other.RawFolder && Destination == other.Destination && LinksWiki == other.LinksWiki
        && LinksWebPartId == other.LinksWebPartId && PublishId == other.PublishId
        && LinksWebPartFolder == other.LinksWebPartFolder && LinksWebPartPage == other.LinksWebPartPage
        && (QuantPages ?? Array.Empty<QuantPage>()).SequenceEqual(other.QuantPages ?? Array.Empty<QuantPage>());

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Server);
        hash.Add(QcWiki);
        hash.Add(QuantWiki);
        hash.Add(RawFolder);
        hash.Add(Destination);
        hash.Add(LinksWiki);
        hash.Add(LinksWebPartId);
        hash.Add(PublishId);
        hash.Add(LinksWebPartFolder);
        hash.Add(LinksWebPartPage);
        return hash.ToHashCode();
    }

    /// <summary>The QC page to publish to: the remembered one, else the default name.</summary>
    public string QcPageFor(string outputDir) => QcWiki?.Page ?? PanoramaPublisher.DefaultQcPage(outputDir);

    /// <summary>
    /// The quant page to publish to. A remembered name PRISM generated follows the CURRENT contrast,
    /// since the quant report in an output directory is replaced by the next contrast analyzed, and
    /// publishing a new contrast under the old one's page would overwrite that page. A name someone
    /// chose is kept as given.
    /// </summary>
    /// <remarks>
    /// Whether a name was generated is decided by <see cref="QuantPages"/>, which records the contrast
    /// each page was published for: generated means it is exactly that contrast's default. A name made
    /// by editing the default - the natural way to choose one - still starts like a default, so the
    /// prefix alone would throw it away; the prefix decides only for a page with no record.
    /// </remarks>
    public string QuantPageFor(string outputDir)
    {
        var current = PanoramaPublisher.DefaultQuantPage(outputDir, PanoramaPublisher.QuantContrast(outputDir));
        if (QuantWiki?.Page is not { } page)
            return current;
        var record = QuantPages?.LastOrDefault(p => p.Page == page);
        var generated = record is not null
            ? page == PanoramaPublisher.DefaultQuantPage(outputDir, record.Contrast)
            : page.StartsWith(PanoramaPublisher.DefaultQuantPage(outputDir, null), StringComparison.Ordinal);
        return generated ? current : page;
    }

    [JsonIgnore]
    public Uri ServerUri => Uri.TryCreate(Server, UriKind.Absolute, out var uri) ? uri : PanoramaPaths.DefaultServer;
}
