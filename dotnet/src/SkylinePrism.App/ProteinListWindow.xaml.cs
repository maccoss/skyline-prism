using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows.Data;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Media;
using Microsoft.Win32;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.App;

/// <summary>
/// Editor for the protein lists (Dynamic Range tab) - the user's own and the ones PRISM ships. Edits a
/// COPY and only hands it back on OK, so Cancel genuinely discards - these lists are persisted per user
/// and shared across every project, which makes an accidental edit expensive.
/// </summary>
public partial class ProteinListWindow : Window
{
    /// <summary>Palette offered for list colors: the categorical set used elsewhere in the tool.</summary>
    private static readonly (string Name, string Hex)[] Palette =
    {
        ("Red", "#d62728"), ("Blue", "#1f77b4"), ("Green", "#2ca02c"), ("Orange", "#ff7f0e"),
        ("Purple", "#9467bd"), ("Brown", "#8c564b"), ("Pink", "#e377c2"), ("Olive", "#bcbd22"),
        ("Teal", "#17becf"), ("Grey", "#7f7f7f"),
    };

    private readonly ObservableCollection<ListRow> _rows = new();     // the user's own
    private readonly ObservableCollection<ListRow> _shipped = new();  // the panels PRISM ships
    private ListRow? _current;
    private bool _suppress;

    public ProteinListWindow(ProteinListSet source)
    {
        InitializeComponent();

        // Two collections, not one, because the two are governed differently: a shipped panel is
        // read-only so that it means the same thing on every machine - which is what makes it citable -
        // while the user's own are freely editable. Mixing them in one box was how a saved list came to
        // silently stand in for a shipped panel of the same name.
        foreach (var list in source.Lists)
            _rows.Add(new ListRow(list.Clone()));

        var mine = source.Lists.Select(l => l.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var list in source.WithBuiltIns())
            if (!mine.Contains(list.Name))
                _shipped.Add(new ListRow(list.Clone()));

        ListsBox.ItemsSource = _rows;

        // 65 shipped panels is not a list anyone reads top to bottom, so group them under collapsible
        // category headings. The user's own stay flat - a handful needs no navigation, and giving them
        // categories would mean asking for one every time a list is created.
        var shippedView = new CollectionViewSource { Source = _shipped };
        shippedView.GroupDescriptions.Add(new PropertyGroupDescription(nameof(ListRow.Category)));
        // ShippedBox sets IsSynchronizedWithCurrentItem="False" explicitly: its default (null) means
        // "synchronize IF the source is an ICollectionView", so handing it a view - as of this change -
        // would silently start selecting the view's current item, which begins at row 0. The two boxes
        // arbitrate selection between themselves below; WPF currency doing it as well is one owner too
        // many.
        ShippedBox.ItemsSource = shippedView.View;
        ColorCombo.ItemsSource = Palette.Select(p => new ColorChoice(p.Name, p.Hex)).ToList();
        if (_rows.Count > 0)
            ListsBox.SelectedIndex = 0;
        else if (_shipped.Count > 0)
            ListTabs.SelectedIndex = 1;
    }

    /// <summary>The edited set; only meaningful when ShowDialog returned true.</summary>
    public ProteinListSet Result { get; private set; } = new();

    /// <summary>Whether the selected row is a shipped panel, which is read-only.</summary>
    private bool CurrentIsShipped => _current is not null && _shipped.Contains(_current);

    private void OnAddList(object sender, RoutedEventArgs e)
    {
        var used = _rows.Select(r => r.Model.ColorHex).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var color = Palette.FirstOrDefault(p => !used.Contains(p.Hex)).Hex ?? Palette[0].Hex;
        var row = new ListRow(new ProteinList { Name = UniqueName($"List {_rows.Count + 1}"), ColorHex = color });
        _rows.Add(row);
        ListTabs.SelectedIndex = 0;
        ListsBox.SelectedItem = row;
        NameBox.Focus();
        NameBox.SelectAll();
    }

    private void OnRemoveList(object sender, RoutedEventArgs e)
    {
        if (ListsBox.SelectedItem is not ListRow row)
            return;
        var index = _rows.IndexOf(row);
        _rows.Remove(row);
        if (_rows.Count > 0)
            ListsBox.SelectedIndex = Math.Min(index, _rows.Count - 1);
        else
            BindDetail(null);
    }

    private void OnSelectedListChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        // Whichever box raised it owns the selection; clear the other so one row is current at a time.
        if (ReferenceEquals(sender, ListsBox) && ListsBox.SelectedItem is not null)
            ShippedBox.SelectedItem = null;
        else if (ReferenceEquals(sender, ShippedBox) && ShippedBox.SelectedItem is not null)
            ListsBox.SelectedItem = null;

        BindDetail((ListsBox.SelectedItem ?? ShippedBox.SelectedItem) as ListRow);
    }

    private void OnListTabChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (!ReferenceEquals(sender, ListTabs) || !IsInitialized)
            return;
        // Bind whatever the newly shown tab has selected, so the detail pane never describes a row the
        // user can no longer see. My lists opens on its first row; the Predefined tab does NOT, because
        // its groups start collapsed - selecting into one would fill the detail pane with a panel that
        // has no visible row, which is the very thing this is here to prevent. Its headings are the
        // starting point there.
        var box = ListTabs.SelectedIndex == 0 ? ListsBox : ShippedBox;
        if (ReferenceEquals(box, ListsBox) && box.SelectedItem is null && box.Items.Count > 0)
            box.SelectedIndex = 0;
        BindDetail(box.SelectedItem as ListRow);
    }

    /// <summary>
    /// Copy the selected shipped panel into My lists, where it can be edited. The shipped one is left
    /// exactly as it was - that is the whole point of the split.
    /// </summary>
    private void OnDuplicateShipped(object sender, RoutedEventArgs e)
    {
        if (ShippedBox.SelectedItem is not ListRow row)
            return;

        var copy = row.Model.Clone();
        copy.Name = UniqueName(copy.Name);
        copy.Visible = false; // a new list starts hidden, like every other list PRISM adds
        copy.Category = "";   // My lists is flat; a heading carried over here is stored and never read
        var added = new ListRow(copy);
        _rows.Add(added);
        ListTabs.SelectedIndex = 0;
        ListsBox.SelectedItem = added;
        NameBox.Focus();
        NameBox.SelectAll();
    }

    /// <summary>A name not already taken by one of the user's lists.</summary>
    private string UniqueName(string wanted)
    {
        bool Taken(string n) =>
            _rows.Any(r => string.Equals(r.Model.Name, n, StringComparison.OrdinalIgnoreCase));
        if (!Taken(wanted))
            return wanted;
        for (var i = 2; ; i++)
        {
            var candidate = $"{wanted} ({i})";
            if (!Taken(candidate))
                return candidate;
        }
    }

    private void BindDetail(ListRow? row)
    {
        _suppress = true;
        try
        {
            _current = row;
            DetailPanel.IsEnabled = row is not null;

            // A shipped panel is shown in full but cannot be altered. Its Visible/ShowLabels ticks stay
            // live: those are the user's view of it, not part of its definition.
            var shipped = row is not null && _shipped.Contains(row);
            ReadOnlyNote.Visibility = shipped ? Visibility.Visible : Visibility.Collapsed;
            NameBox.IsReadOnly = shipped;
            MembersBox.IsReadOnly = shipped;
            ColorCombo.IsEnabled = !shipped;
            ImportButton.IsEnabled = !shipped;

            NameBox.Text = row?.Model.Name ?? "";
            MembersBox.Text = row is null ? "" : string.Join(Environment.NewLine, row.Model.Members);
            ShowLabelsCheck.IsChecked = row?.Model.ShowLabels ?? false;
            ColorCombo.SelectedItem = row is null
                ? null
                : (ColorCombo.ItemsSource as IEnumerable<ColorChoice>)?.FirstOrDefault(
                    c => string.Equals(c.Hex, row.Model.ColorHex, StringComparison.OrdinalIgnoreCase));
        }
        finally
        {
            _suppress = false;
        }
    }

    private void OnNameChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (_suppress || _current is null)
            return;
        _current.Model.Name = NameBox.Text;
        _current.Refresh();
    }

    private void OnMembersChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (_suppress || _current is null)
            return;
        _current.Model.Members = MembersBox.Text
            .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .SelectMany(ProteinList.SplitMemberLine)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        _current.Refresh();
    }

    private void OnColorChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (_suppress || _current is null || ColorCombo.SelectedItem is not ColorChoice choice)
            return;
        _current.Model.ColorHex = choice.Hex;
        _current.Refresh();
    }

    private void OnShowLabelsChanged(object sender, RoutedEventArgs e)
    {
        if (_suppress || _current is null)
            return;
        _current.Model.ShowLabels = ShowLabelsCheck.IsChecked == true;
    }

    private void OnImportMembers(object sender, RoutedEventArgs e)
    {
        if (_current is null)
            return;
        var dialog = new OpenFileDialog
        {
            Title = "Import protein list members",
            Filter = "Text or CSV (*.txt;*.csv;*.tsv)|*.txt;*.csv;*.tsv|All files (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) != true)
            return;

        try
        {
            var imported = ProteinListSet.ReadMembersFile(dialog.FileName);
            // Merge rather than replace: importing a second file extends the list.
            var merged = _current.Model.Members.Concat(imported)
                .Distinct(StringComparer.OrdinalIgnoreCase).ToList();
            _current.Model.Members = merged;
            if (string.IsNullOrWhiteSpace(_current.Model.Name) || _current.Model.Name.StartsWith("List "))
                _current.Model.Name = System.IO.Path.GetFileNameWithoutExtension(dialog.FileName);
            BindDetail(_current);
            _current.Refresh();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Could not import the list",
                MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void OnOk(object sender, RoutedEventArgs e)
    {
        // Only the user's lists are saved as lists. A shipped panel contributes just its two view flags,
        // so ticking one never forks its membership and it still picks up a later correction.
        var result = new ProteinListSet { Lists = _rows.Select(r => r.Model).ToList() };
        foreach (var row in _shipped)
            result.SetShippedState(row.Model.Name, row.Model.Visible, row.Model.ShowLabels);
        Result = result;
        DialogResult = true;
    }

    /// <summary>One row in the lists box; wraps the model so the UI can bind swatch/count.</summary>
    private sealed class ListRow : INotifyPropertyChanged
    {
        public ListRow(ProteinList model) => Model = model;

        public ProteinList Model { get; }

        public string Name => Model.Name;
        public string Category => string.IsNullOrWhiteSpace(Model.Category) ? "Other" : Model.Category;
        public string CountLabel => $"({Model.Members.Count})";
        public Brush Brush => ColorChoice.BrushFor(Model.ColorHex);

        public bool Visible
        {
            get => Model.Visible;
            set
            {
                Model.Visible = value;
                Raise(nameof(Visible));
            }
        }

        public void Refresh()
        {
            Raise(nameof(Name));
            Raise(nameof(CountLabel));
            Raise(nameof(Brush));
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        private void Raise([CallerMemberName] string? name = null)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }

    private sealed record ColorChoice(string Name, string Hex)
    {
        public Brush Brush => BrushFor(Hex);

        public static Brush BrushFor(string hex)
        {
            try
            {
                var brush = new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
                brush.Freeze();
                return brush;
            }
            catch (FormatException)
            {
                return Brushes.Gray;
            }
        }
    }
}
