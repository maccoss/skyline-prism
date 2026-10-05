using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Documents;
using SkylinePrism.Core.Panorama;

namespace SkylinePrism.App;

/// <summary>
/// The Skyline tool's Publish to Panorama window: the GUI over <see cref="OutputPublishing"/>, the same
/// publish <c>prism publish</c> runs, so the two cannot disagree. Show Command Line gives that command.
/// </summary>
/// <remarks>
/// <para>The targets are read from and saved to the output directory's <c>panorama.json</c>, so the
/// window opens on the folders this output directory was last published to - and so does the CLI.</para>
/// <para>Sign-in is attempted silently on open, with the sign-ins already saved on this computer
/// (PanoramaBridge's first). The sign-in window appears only when none of them works.</para>
/// </remarks>
public partial class PanoramaPublishWindow : Window
{
    private readonly string _outputDir;
    private PanoramaClient? _client;
    private CancellationTokenSource? _running;
    private bool _loading = true;

    // The links page goes in the QC page's folder unless told otherwise - the quant page's, without a QC
    // page - as in the CLI: the folder box follows the reports' folders until someone types in it, or
    // until this directory remembers its own.
    private bool _linksFollowReports;
    private bool _settingLinksFolder;

    // Each quiet sign-in's number: one overtaken by a later sign-in (the server changed meanwhile)
    // must neither replace the newer client nor leak its own.
    private int _signInAttempt;

    internal PanoramaPublishWindow(string outputDir)
    {
        InitializeComponent();
        _outputDir = outputDir;
        var targets = PanoramaTargets.Load(outputDir);

        ServerBox.Text = targets.ServerUri.GetLeftPart(UriPartial.Authority);
        QcFolderBox.Text = targets.QcWiki?.Folder ?? "";
        QcPageBox.Text = targets.QcPageFor(outputDir);
        QuantFolderBox.Text = targets.QuantWiki?.Folder ?? targets.QcWiki?.Folder ?? "";
        QuantPageBox.Text = targets.QuantPageFor(outputDir);
        RawFolderBox.Text = targets.RawFolder ?? "";
        _linksFollowReports = targets.LinksWiki is null;
        LinksFolderBox.Text = targets.LinksWiki?.Folder ?? targets.QcWiki?.Folder ?? targets.QuantWiki?.Folder ?? "";
        LinksPageBox.Text = targets.LinksPageFor(outputDir);

        var request = new PublishRequest(outputDir, targets.ServerUri, null, null, null);
        QcCheck.IsChecked = File.Exists(request.QcReport);
        QcCheck.IsEnabled = File.Exists(request.QcReport);
        var hasQuant = File.Exists(request.QuantReport);
        QuantCheck.IsChecked = hasQuant;
        QuantCheck.IsEnabled = hasQuant;
        QuantContrastText.Text = hasQuant
            ? $"Contrast: {PanoramaPublisher.QuantContrast(outputDir) ?? "(not recorded)"}"
            : "This output directory has no quant report yet - make one with the Differential pane's Quant report button.";
        UploadCheck.IsChecked = true;
        LinksCheck.IsChecked = true;

        _loading = false;
        Refresh();
        Loaded += async (_, _) => await SignInQuietlyAsync();
        Closing += (_, e) =>
        {
            if (_running is not null)
            {
                if (MessageBox.Show(this, "Stop publishing? Files already uploaded stay on Panorama, and publishing again skips them.",
                        Title, MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes)
                {
                    e.Cancel = true;
                    return;
                }

                _running.Cancel();
            }
        };
        Closed += (_, _) => _client?.Dispose();
    }

    /// <summary>Opens the window for an output directory.</summary>
    public static void Open(Window owner, string outputDir) =>
        new PanoramaPublishWindow(outputDir) { Owner = owner }.ShowDialog();

    private Uri Server =>
        Uri.TryCreate(ServerBox.Text.Contains("://", StringComparison.Ordinal) ? ServerBox.Text.Trim() : "https://" + ServerBox.Text.Trim(),
            UriKind.Absolute, out var uri)
            ? uri
            : PanoramaPaths.DefaultServer;

    /// <summary>The publish the window currently describes.</summary>
    private PublishRequest CurrentRequest()
    {
        string? Folder(bool on, string text) => on && !string.IsNullOrWhiteSpace(text) ? PanoramaPaths.FromAnything(text) : null;
        string? Page(string text) => string.IsNullOrWhiteSpace(text) ? null : text.Trim();
        return new PublishRequest(_outputDir, Server,
            Folder(QcCheck.IsChecked == true, QcFolderBox.Text),
            Folder(QuantCheck.IsChecked == true, QuantFolderBox.Text),
            Folder(UploadCheck.IsChecked == true, RawFolderBox.Text),
            QcPage: Page(QcPageBox.Text),
            QuantPage: Page(QuantPageBox.Text),
            ReplaceEdited: ReplaceEditedCheck.IsChecked == true,
            LinksFolder: Folder(LinksCheck.IsChecked == true, LinksFolderBox.Text),
            LinksPage: Page(LinksPageBox.Text));
    }

    private void OnTargetsChanged(object sender, RoutedEventArgs e)
    {
        var reportFolderChanged = ReferenceEquals(sender, QcFolderBox) || ReferenceEquals(sender, QuantFolderBox)
                                  || ReferenceEquals(sender, QcCheck) || ReferenceEquals(sender, QuantCheck);
        if (!_loading && _linksFollowReports && reportFolderChanged && FollowedLinksFolder() is { } follow
            && follow != LinksFolderBox.Text)
        {
            _settingLinksFolder = true;
            LinksFolderBox.Text = follow;
            _settingLinksFolder = false;
        }

        Refresh();
    }

    /// <summary>
    /// The CLI's default links folder: the QC page's folder, else the quant page's; null (leave the box
    /// alone) when neither report is being published to a folder.
    /// </summary>
    private string? FollowedLinksFolder() =>
        QcCheck.IsChecked == true && !string.IsNullOrWhiteSpace(QcFolderBox.Text) ? QcFolderBox.Text
        : QuantCheck.IsChecked == true && !string.IsNullOrWhiteSpace(QuantFolderBox.Text) ? QuantFolderBox.Text
        : null;

    private void OnLinksFolderChanged(object sender, RoutedEventArgs e)
    {
        // Typed (or browsed) by someone: from now on it is theirs, not the reports' folders'.
        if (!_loading && !_settingLinksFolder)
            _linksFollowReports = false;
        Refresh();
    }

    /// <summary>Re-reads the targets: the destination line, the plan, and whether Publish can run.</summary>
    private void Refresh()
    {
        if (_loading || _running is not null)
            return;

        var request = CurrentRequest();
        var problems = OutputPublishing.Problems(request);
        // A checked step with no folder yet is said once, not reported as a refusal.
        var missing = (QcCheck.IsChecked == true && request.QcFolder is null ? "Choose a folder for the QC report's wiki page. " : "")
                      + (QuantCheck.IsChecked == true && request.QuantFolder is null ? "Choose a folder for the quant report's wiki page. " : "")
                      + (UploadCheck.IsChecked == true && request.RawFolder is null ? "Choose the folder that holds the raw files. " : "")
                      + (LinksCheck.IsChecked == true && request.LinksFolder is null ? "Choose a folder for the links page. " : "");

        DestinationText.Text = request.RawFolder is not null && PanoramaPaths.IsFileArea(request.RawFolder)
            ? $"Uploads to {request.Destination}"
            : "The output directory goes beside this folder, under its own name.";

        LogBox.Text = problems.Count > 0 || missing.Length > 0
            ? (missing + string.Join(Environment.NewLine, problems)).Trim()
            : "Will:" + Environment.NewLine + OutputPublishing.Describe(request);
        PublishButton.IsEnabled = _client is not null && problems.Count == 0 && missing.Length == 0 && request.HasWork;
    }

    private void OnServerChanged(object sender, RoutedEventArgs e)
    {
        if (_client is not null && !string.Equals(_client.Server.Host, Server.Host, StringComparison.OrdinalIgnoreCase))
        {
            _client.Dispose();
            _client = null;
            _ = SignInQuietlyAsync();
        }
    }

    private async Task SignInQuietlyAsync()
    {
        var attempt = ++_signInAttempt;
        var server = Server;
        SignInText.Text = $"Signing in to {server.Host}...";
        try
        {
            var client = await OutputPublishing.SignInAsync(PanoramaSignIn.ForThisComputer(server));
            if (attempt != _signInAttempt)
            {
                client.Dispose(); // overtaken by a sign-in to the server now in the box
                return;
            }

            _client?.Dispose();
            _client = client;
            SignInText.Text = $"Signed in with the sign-in from {client.Credential.Source} ({client.Credential}).";
            SignInButton.Content = "Change...";
        }
        catch (Exception ex)
        {
            // Called from the Loaded handler, which has no caller to propagate to.
            if (attempt != _signInAttempt)
                return;
            SignInText.Text = ex.Message;
            SignInButton.Content = "Sign in...";
        }

        Refresh();
    }

    private void OnSignIn(object sender, RoutedEventArgs e)
    {
        var why = _client is null
            ? "No saved sign-in for Panorama works on this computer. Sign in once and PRISM keeps it for next time."
            : $"Signed in now with the sign-in from {_client.Credential.Source}. Sign in with another one to use it instead.";
        if (PanoramaSignInWindow.Ask(this, PanoramaSignIn.ForThisComputer(Server), why) is not { } client)
            return;
        _signInAttempt++; // a quiet sign-in still running must not replace this one
        _client?.Dispose();
        _client = client;
        SignInText.Text = $"Signed in with {client.Credential}{(client.Credential.Source == "typed" ? "" : $", saved by {client.Credential.Source}")}.";
        SignInButton.Content = "Change...";
        Refresh();
    }

    private void OnBrowseQc(object sender, RoutedEventArgs e) =>
        Browse(QcFolderBox, "Choose the Panorama folder whose wiki gets the QC report.", fileAreaOnly: false);

    private void OnBrowseQuant(object sender, RoutedEventArgs e) =>
        Browse(QuantFolderBox, "Choose the Panorama folder whose wiki gets the quant report.", fileAreaOnly: false);

    private void OnBrowseLinks(object sender, RoutedEventArgs e) =>
        Browse(LinksFolderBox, "Choose the Panorama folder whose page shows the links to this output directory's results.", fileAreaOnly: false);

    private void OnBrowseRaw(object sender, RoutedEventArgs e) =>
        Browse(RawFolderBox, "Choose the folder that holds this experiment's raw files (inside @files). The output directory is uploaded next to it.",
            fileAreaOnly: true);

    private void Browse(System.Windows.Controls.TextBox box, string prompt, bool fileAreaOnly)
    {
        if (_client is null)
        {
            MessageBox.Show(this, "Sign in to Panorama first.", Title, MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var start = string.IsNullOrWhiteSpace(box.Text) ? null : PanoramaPaths.FromAnything(box.Text);
        if (PanoramaFolderWindow.Ask(this, _client, prompt, start, fileAreaOnly) is { } folder)
            box.Text = folder;
    }

    private async void OnPublish(object sender, RoutedEventArgs e)
    {
        if (_running is not null)
        {
            _running.Cancel();
            return;
        }

        if (_client is not { } client)
            return;

        var request = CurrentRequest();
        _running = new CancellationTokenSource();
        SetEditable(false);
        PublishButton.Content = "Stop";
        ResultsText.Inlines.Clear();
        LogBox.Text = "";
        UploadProgress.Value = 0;
        UploadProgress.Visibility = request.RawFolder is null ? Visibility.Collapsed : Visibility.Visible;

        var progress = new Progress<UploadProgress>(p =>
        {
            UploadProgress.Maximum = Math.Max(1, p.TotalBytes);
            UploadProgress.Value = p.TotalSent;
            UploadProgress.ToolTip = $"{p.File}: file {p.FileIndex} of {p.FileCount}, "
                                     + $"{PanoramaPublisher.FormatBytes(p.TotalSent)} of {PanoramaPublisher.FormatBytes(p.TotalBytes)}";
        });

        try
        {
            var outcome = await Task.Run(() => OutputPublishing.RunAsync(client, request,
                say: m => Dispatcher.BeginInvoke(new Action(() => Log(m))), progress, _running.Token));
            Log("Done.");
            if (outcome.Upload is { } up)
                AddLink($"Output directory ({up.Uploaded} uploaded, {up.Skipped} already there)", up.Url);
            if (outcome.Qc is { } qc)
                AddLink($"QC report wiki page ({(qc.Created ? "new" : "updated")})", qc.Url);
            if (outcome.Quant is { } quant)
                AddLink($"Quant report wiki page ({(quant.Created ? "new" : "updated")})", quant.Url);
            if (outcome.Links is { } links)
                AddLink(outcome.LinksOnFolderPage
                        ? $"Folder page, with the links ({links.Folder})"
                        : $"Links page ({(links.Created ? "new" : "updated")}; not on the folder's page - see the log)",
                    outcome.LinksOnFolderPage ? PanoramaPaths.BrowserUrl(links.Folder, client.Server) : links.Url);
        }
        catch (OperationCanceledException)
        {
            Log("Stopped. Files already uploaded stay on Panorama; publishing again skips them.");
        }
        catch (PanoramaException ex)
        {
            Log("Not published: " + ex.Message);
        }
        catch (Exception ex)
        {
            // async void: nothing above this can catch it, and an escaping exception would take the
            // tool down with a modal dialog (UiThreadSafetyTests).
            Log($"Not published - something unexpected went wrong: {ex.Message}");
        }
        finally
        {
            _running.Dispose();
            _running = null;
            PublishButton.Content = "Publish";
            SetEditable(true);
            UploadProgress.Visibility = Visibility.Collapsed;
        }
    }

    private void SetEditable(bool editable)
    {
        foreach (var control in new System.Windows.Controls.Control[]
                 {
                     ServerBox, SignInButton, QcCheck, QcFolderBox, QcPageBox, QuantFolderBox, QuantPageBox, UploadCheck, RawFolderBox,
                     LinksCheck, LinksFolderBox, LinksPageBox, ReplaceEditedCheck, CommandButton,
                 })
            control.IsEnabled = editable;
        QuantCheck.IsEnabled = editable && File.Exists(Path.Combine(_outputDir, "quant", "quant_report.html"));
    }

    private void Log(string line)
    {
        LogBox.AppendText((LogBox.Text.Length > 0 ? Environment.NewLine : "") + line);
        LogBox.ScrollToEnd();
    }

    private void AddLink(string label, string url)
    {
        if (ResultsText.Inlines.Count > 0)
            ResultsText.Inlines.Add(new LineBreak());
        var link = new Hyperlink(new Run(label)) { NavigateUri = new Uri(url), ToolTip = url };
        link.RequestNavigate += (_, args) => Process.Start(new ProcessStartInfo(args.Uri.AbsoluteUri) { UseShellExecute = true });
        ResultsText.Inlines.Add(link);
    }

    private void OnShowCommand(object sender, RoutedEventArgs e)
    {
        var command = OutputPublishing.CommandLine(CurrentRequest());
        try
        {
            Clipboard.SetText(command);
        }
        catch (System.Runtime.InteropServices.ExternalException)
        {
            // Another program holds the clipboard; the command is still shown.
        }

        MessageBox.Show(this, command + Environment.NewLine + Environment.NewLine
                              + "(Copied to the clipboard. The CLI signs in the same way: PRISM_PANORAMA_API_KEY if set, "
                              + "otherwise the sign-ins saved on this computer.)",
            "prism publish", MessageBoxButton.OK, MessageBoxImage.Information);
    }

    private void OnClose(object sender, RoutedEventArgs e) => Close();
}
