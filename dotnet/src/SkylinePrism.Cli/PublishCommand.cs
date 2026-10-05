using System;
using System.Collections.Generic;
using System.Threading;
using SkylinePrism.Core.Panorama;

namespace SkylinePrism.Cli;

public static partial class Program
{
    /// <summary>
    /// <c>prism publish</c>: the QC and quant reports as Panorama wiki pages, and the output directory
    /// uploaded beside the raw files - the same publish as the Skyline tool's Publish to Panorama window
    /// (<see cref="OutputPublishing"/>), so a headless run and the GUI cannot disagree.
    /// </summary>
    private static int CmdPublish(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var dir = opts.GetSingleOrNull("-d", "--dir")
                  ?? throw new ArgumentException("prism publish needs -d <output-dir>.");
        var remembered = PanoramaTargets.Load(dir);

        // A folder may be given as a path or pasted as a browser address; either is reduced to the folder.
        string? Folder(string flag) => opts.GetSingleOrNull(flag) is { } v ? PanoramaPaths.FromAnything(v) : null;
        var server = opts.GetSingleOrNull("--server") is { } s
            ? new Uri(s.Contains("://", StringComparison.Ordinal) ? s : "https://" + s)
            : remembered.ServerUri;

        // Flags override what the output directory remembers; --no-upload / --no-qc / --no-quant drop a
        // remembered step for this run without forgetting it.
        var qcFolder = opts.GetSingleOrNull("--no-qc") is not null ? null : Folder("--qc-wiki") ?? remembered.QcWiki?.Folder;
        var quantFolder = opts.GetSingleOrNull("--no-quant") is not null ? null : Folder("--quant-wiki") ?? remembered.QuantWiki?.Folder;
        var rawFolder = opts.GetSingleOrNull("--no-upload") is not null ? null : Folder("--beside-raw") ?? remembered.RawFolder;
        // The links page goes with the wiki pages unless told otherwise: by default in the QC page's
        // folder, whose own page then shows it.
        var linksFolder = opts.GetSingleOrNull("--no-links") is not null
            ? null
            : Folder("--links-wiki") ?? remembered.LinksWiki?.Folder ?? (qcFolder ?? quantFolder);
        var request = new PublishRequest(dir, server, qcFolder, quantFolder, rawFolder,
            QcPage: opts.GetSingleOrNull("--qc-page") ?? remembered.QcPageFor(dir),
            QuantPage: opts.GetSingleOrNull("--quant-page") ?? remembered.QuantPageFor(dir),
            ReplaceEdited: opts.GetSingleOrNull("--replace-edited") is not null,
            TakeOver: opts.GetSingleOrNull("--take-over") is not null,
            LinksFolder: linksFolder,
            LinksPage: opts.GetSingleOrNull("--links-page") ?? remembered.LinksPageFor(dir));

        if (OutputPublishing.Problems(request) is { Count: > 0 } problems)
        {
            foreach (var p in problems)
                Console.Error.WriteLine($"error: {p}");
            return 2;
        }

        Console.WriteLine(OutputPublishing.Describe(request));
        if (opts.GetSingleOrNull("--dry-run") is not null)
        {
            Console.WriteLine("(dry run: nothing was sent)");
            return 0;
        }

        using var cancel = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cancel.Cancel();
        };

        using var client = OutputPublishing.SignInAsync(PanoramaSignIn.ForThisComputer(server), cancel.Token).GetAwaiter().GetResult();
        Console.WriteLine($"Signed in to {server.Host} with the sign-in from {client.Credential.Source} ({client.Credential}).");

        // One line per tenth of the upload, not one per buffer: a log, not a progress bar.
        long nextReport = 0;
        var progress = new SynchronousProgress<UploadProgress>(p =>
        {
            if (p.TotalBytes == 0 || p.TotalSent < nextReport)
                return;
            nextReport = p.TotalSent + p.TotalBytes / 10;
            Console.WriteLine($"  {PanoramaPublisher.FormatBytes(p.TotalSent)} of {PanoramaPublisher.FormatBytes(p.TotalBytes)} "
                              + $"(file {p.FileIndex} of {p.FileCount})");
        });

        var outcome = OutputPublishing.RunAsync(client, request, say: m => Console.WriteLine("  " + m), progress, cancel.Token)
            .GetAwaiter().GetResult();

        if (outcome.Upload is { } up)
            Console.WriteLine($"Uploaded {up.Uploaded} files ({PanoramaPublisher.FormatBytes(up.BytesSent)}), "
                              + $"{up.Skipped} already there: {up.Url}");
        if (outcome.Qc is { } qc)
            Console.WriteLine($"QC report {(qc.Created ? "published" : "updated")}: {qc.Url}");
        if (outcome.Quant is { } quant)
            Console.WriteLine($"Quant report {(quant.Created ? "published" : "updated")}: {quant.Url}");
        if (outcome.Links is { } links)
            Console.WriteLine(outcome.LinksOnFolderPage
                ? $"Links shown on {links.Folder}'s page: {PanoramaPaths.BrowserUrl(links.Folder, server)}"
                : $"Links page published (not on the folder's page; see above): {links.Url}");
        Console.WriteLine($"Targets saved in {System.IO.Path.Combine(dir, PanoramaTargets.FileName)}; "
                          + "'prism publish -d <dir>' alone republishes to them.");
        return 0;
    }

    /// <summary>
    /// Reports on the calling thread. <see cref="Progress{T}"/> posts to the thread pool from a
    /// console app, so its lines would arrive out of order with the ones written directly.
    /// </summary>
    private sealed class SynchronousProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }

    private const string PublishHelp = """
        prism publish - Publish an output directory to Panorama

        Publishes the QC report and the quant report as wiki pages on Panorama, each in a
        folder you choose, and uploads the whole output directory beside the folder that
        holds the raw files. The same publish as the Skyline tool's Publish to Panorama
        window.

        Usage: prism publish -d <output-dir> [options]

        Options:
            -d, --dir DIR          Output directory from `prism run`
            --qc-wiki FOLDER       Panorama folder for the QC report's wiki page
            --quant-wiki FOLDER    Panorama folder for the quant report's wiki page
                                   (quant/quant_report.html, from `prism differential --report`)
            --beside-raw FOLDER    The Panorama folder holding the raw files; the output
                                   directory is uploaded next to it, under its own name
                                   (e.g. .../@files/RawFiles -> .../@files/<output-dir name>)
            --links-wiki FOLDER    Panorama folder for the links page - links to the QC page, every
                                   quant page published from this directory, and the uploaded
                                   files - which that folder's own page then shows in a Wiki web
                                   part, right after Targeted MS Runs (so above Files). Default:
                                   the QC page's folder. Adding the web part needs folder
                                   administrator permission; without it the links page is still
                                   published and the output says so
            --qc-page NAME         Wiki page name (default PRISM-QC-<output-dir name>)
            --quant-page NAME      Wiki page name (default PRISM-Quant-<output-dir name>-<contrast>)
            --links-page NAME      Links page name (default PRISM-<output-dir name>)
            --server URL           Panorama server (default https://panoramaweb.org)
            --replace-edited       Replace a page edited on Panorama since PRISM published
                                   it, or a page of that name PRISM did not write. Without
                                   it, either is refused rather than overwritten
            --take-over            Replace a page PRISM published from a different output
                                   directory - another analysis whose directory name gives
                                   the same page name, or this one if its panorama.json was
                                   lost. Separate from --replace-edited on purpose
            --no-upload, --no-qc, --no-quant, --no-links
                                   Skip a step remembered from an earlier publish, this time
            --dry-run              Print what would be published, and send nothing

        A FOLDER may be a path (/MacCoss/maccoss/Project/@files/RawFiles) or an address
        pasted from a browser. Every publish records its targets in <output-dir>/panorama.json,
        so `prism publish -d <dir>` alone republishes to the same places.

        Republishing updates each page in place (Panorama keeps the earlier versions of its
        text, and only changed plots are sent) and uploads only the files that changed: each
        file is compared with Panorama's own MD5 of its copy, and every upload is verified the
        same way.

        Sign-in: an API key in PRISM_PANORAMA_API_KEY if set, otherwise the sign-in
        PanoramaBridge saved on this computer, then LabOps's, then the one saved by the
        Skyline tool. Nothing is written to the output directory but folder and page names.
        """;
}
