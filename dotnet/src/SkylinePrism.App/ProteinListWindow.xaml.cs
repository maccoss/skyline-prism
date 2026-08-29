using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
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

    private readonly ObservableCollection<ListRow> _rows = new();
    private ListRow? _current;
    private bool _suppress;

    public ProteinListWindow(ProteinListSet source)
    {
        InitializeComponent();

        // Seeded from WithBuiltIns, not from Lists: the panels PRISM ships (EV markers, Glomerulus,
        // Tubular contamination) show up here alongside the user's own, so they can be ticked on for the
        // plot or edited into a cohort-specific variant. They arrive unticked, and a name the user has
        // already defined wins - so an override replaces the shipped one rather than doubling it.
        foreach (var list in source.WithBuiltIns())
            _rows.Add(new ListRow(list.Clone()));
        ListsBox.ItemsSource = _rows;
        ColorCombo.ItemsSource = Palette.Select(p => new ColorChoice(p.Name, p.Hex)).ToList();
        if (_rows.Count > 0)
            ListsBox.SelectedIndex = 0;
    }

    /// <summary>The edited set; only meaningful when ShowDialog returned true.</summary>
    public ProteinListSet Result { get; private set; } = new();

    private void OnAddList(object sender, RoutedEventArgs e)
    {
        var used = _rows.Select(r => r.Model.ColorHex).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var color = Palette.FirstOrDefault(p => !used.Contains(p.Hex)).Hex ?? Palette[0].Hex;
        var row = new ListRow(new ProteinList { Name = $"List {_rows.Count + 1}", ColorHex = color });
        _rows.Add(row);
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
        => BindDetail(ListsBox.SelectedItem as ListRow);

    private void BindDetail(ListRow? row)
    {
        _suppress = true;
        try
        {
            _current = row;
            DetailPanel.IsEnabled = row is not null;
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
            .Split(new[] { '\r', '\n', ',', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(m => m.Trim())
            .Where(m => m.Length > 0)
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
        Result = new ProteinListSet { Lists = _rows.Select(r => r.Model).ToList() };
        DialogResult = true;
    }

    /// <summary>One row in the lists box; wraps the model so the UI can bind swatch/count.</summary>
    private sealed class ListRow : INotifyPropertyChanged
    {
        public ListRow(ProteinList model) => Model = model;

        public ProteinList Model { get; }

        public string Name => Model.Name;
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
