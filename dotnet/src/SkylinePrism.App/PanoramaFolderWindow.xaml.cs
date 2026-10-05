using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using SkylinePrism.Core.Panorama;

namespace SkylinePrism.App;

/// <summary>
/// Browses Panorama's folders, one level at a time, and returns the one chosen.
/// </summary>
/// <remarks>
/// <para>Children are fetched when a folder is first expanded rather than up front: the MacCoss project
/// alone has around sixty sub-folders, each with its own tree, so eager loading would mean thousands of
/// requests to show one level (PanoramaBridge's remote browser works the same way, for the same
/// reason).</para>
/// <para>The tree opens on the folder already chosen - or the MacCoss project - with its parents
/// expanded, so the usual case is one click.</para>
/// </remarks>
public partial class PanoramaFolderWindow : Window
{
    private readonly PanoramaClient _client;
    private readonly bool _fileAreaOnly;

    private PanoramaFolderWindow(PanoramaClient client, string prompt, string? start, bool fileAreaOnly)
    {
        InitializeComponent();
        _client = client;
        _fileAreaOnly = fileAreaOnly;
        PromptText.Text = prompt;
        PathBox.Text = start ?? "";
        Loaded += async (_, _) => await OpenAtAsync(string.IsNullOrWhiteSpace(start) ? PanoramaPaths.StartFolder : start);
    }

    /// <summary>The folder chosen, as a folder path, or null when the person canceled.</summary>
    public string? Chosen { get; private set; }

    /// <summary>Shows the browser; <paramref name="fileAreaOnly"/> refuses a folder outside <c>@files</c>.</summary>
    public static string? Ask(Window owner, PanoramaClient client, string prompt, string? start, bool fileAreaOnly)
    {
        var window = new PanoramaFolderWindow(client, prompt, start, fileAreaOnly) { Owner = owner };
        return window.ShowDialog() == true ? window.Chosen : null;
    }

    /// <summary>Loads the projects, then expands down to <paramref name="folder"/> as far as it exists.</summary>
    private async Task OpenAtAsync(string folder)
    {
        StatusText.Text = "Reading Panorama's folders...";
        var items = FolderTree.Items;
        TreeViewItem? last = null;
        var path = "/";
        try
        {
            await FillAsync(items, path);
            foreach (var segment in PanoramaPaths.ToFolder(folder).Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries))
            {
                path = PanoramaPaths.Combine(path, segment);
                var next = items.OfType<TreeViewItem>().FirstOrDefault(i => string.Equals((string)i.Tag, path, StringComparison.OrdinalIgnoreCase));
                if (next is null)
                    break;
                last = next;
                await ExpandAsync(next);
                items = next.Items;
            }

            StatusText.Text = "";
        }
        catch (Exception ex)
        {
            // Reached from the Loaded handler, which has no caller to propagate to.
            StatusText.Text = ex.Message;
        }

        if (last is not null)
        {
            last.IsSelected = true;
            last.BringIntoView();
        }
    }

    private async Task FillAsync(ItemCollection items, string folder)
    {
        items.Clear();
        var entries = await _client.ListAsync(folder);
        foreach (var entry in entries.Where(e => e.IsFolder).OrderBy(e => e.Name.StartsWith('@') ? 0 : 1).ThenBy(e => e.Name, StringComparer.OrdinalIgnoreCase))
        {
            var item = new TreeViewItem
            {
                Header = entry.Name + (PanoramaPaths.IsFileArea(entry.Folder) && !entry.CanUpload ? "  (read only)" : ""),
                Tag = entry.Folder,
            };
            // A placeholder child makes the folder expandable before its contents are known; WPF only
            // draws an expander on an item that already has a child.
            item.Items.Add(new TreeViewItem { Header = "Loading...", IsEnabled = false });
            item.Expanded += async (s, e) =>
            {
                if (ReferenceEquals(e.OriginalSource, s))
                    await ExpandAsync((TreeViewItem)s);
            };
            items.Add(item);
        }
    }

    private async Task ExpandAsync(TreeViewItem item)
    {
        if (item.Items.Count == 1 && item.Items[0] is TreeViewItem { IsEnabled: false })
        {
            try
            {
                await FillAsync(item.Items, (string)item.Tag);
            }
            catch (Exception ex)
            {
                // Reached from a TreeViewItem's Expanded handler, which has no caller to propagate to.
                item.Items.Clear();
                StatusText.Text = ex.Message;
            }
        }

        item.IsExpanded = true;
    }

    private void OnSelectedFolderChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
    {
        if (e.NewValue is TreeViewItem { Tag: string folder })
            PathBox.Text = folder;
    }

    private void OnChoose(object sender, RoutedEventArgs e)
    {
        var folder = PanoramaPaths.FromAnything(PathBox.Text);
        if (folder.Trim('/').Length == 0)
        {
            StatusText.Text = "Choose a folder in the tree, or type one.";
            return;
        }

        if (_fileAreaOnly && !PanoramaPaths.IsFileArea(folder))
        {
            StatusText.Text = $"{folder} is not a file area. Choose a folder inside @files - the one that holds the raw files.";
            return;
        }

        Chosen = folder;
        DialogResult = true;
    }
}
