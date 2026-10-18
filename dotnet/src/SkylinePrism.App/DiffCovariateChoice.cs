using System;
using System.Windows;
using SkylinePrism.Core.DifferentialAnalysis;

namespace SkylinePrism.App;

/// <summary>
/// One Adjust-for column: whether it is ticked, the type it will be fitted as (inferred by
/// <see cref="CovariateTyping.Infer"/>, switchable), and what the design will do with it.
/// </summary>
/// <remarks>
/// A ticked column gets its own row under the picker, out in the Comparison section rather than
/// folded inside Method, with its type as a two-way toggle and any problem written beside it
/// before anything runs. All of that used to be one word inside a closed drop-down, which is easy
/// to miss - and what it guards against, an ID number fitted as one straight-line effect, changes
/// every p-value without anything else on screen looking different.
/// <para>Its own file, not nested in MainWindow: <c>UiThreadSafetyTests</c> tracks MainWindow's
/// members by name, and a view-model property such as <c>Kind</c> would be mistaken for every other
/// <c>.Kind</c> in the window.</para>
/// </remarks>
internal sealed class DiffCovariateChoice : System.ComponentModel.INotifyPropertyChanged
{
    private bool _isSelected;
    private CovariateKind _kind;

    public required string Name { get; init; }

    /// <summary>The inferred type and why.</summary>
    public required CovariateTypeGuess Inferred { get; init; }

    /// <summary>Raised on a tick, so the rows, the summary text and the method line follow.</summary>
    public Action? Changed { get; init; }

    /// <summary>Raised when the type is switched; that changes the model, so the contrast reruns.</summary>
    public Action? KindChanged { get; init; }

    private string _note = string.Empty;

    /// <summary>What the design will do with this column over the current arms; empty when it goes in as it is.</summary>
    public string Note
    {
        get => _note;
        set
        {
            if (_note == value)
                return;
            _note = value;
            Raise(nameof(Note));
            Raise(nameof(NoteVisibility));
        }
    }

    public Visibility NoteVisibility => string.IsNullOrEmpty(_note) ? Visibility.Collapsed : Visibility.Visible;

    public bool IsSelected
    {
        get => _isSelected;
        set
        {
            if (_isSelected == value)
                return;
            _isSelected = value;
            Raise(nameof(IsSelected));
            Changed?.Invoke();
        }
    }

    public CovariateKind Kind
    {
        get => _kind;
        set
        {
            if (_kind == value)
                return;
            _kind = value;
            Raise(nameof(Kind));
            Raise(nameof(KindIndex));
            Raise(nameof(Reason));
            KindChanged?.Invoke();
        }
    }

    /// <summary>Whether the type differs from the inferred one.</summary>
    public bool Overridden => Kind != Inferred.Kind;

    /// <summary>The segmented toggle's index: 0 numeric, 1 categorical.</summary>
    public int KindIndex
    {
        get => Kind == CovariateKind.Numeric ? 0 : 1;
        set
        {
            if (CanSwitch && value is 0 or 1)
                Kind = value == 0 ? CovariateKind.Numeric : CovariateKind.Categorical;
        }
    }

    /// <summary>Why it has this type, in a line under the toggle.</summary>
    public string Reason => !Inferred.AllNumeric
        ? $"Categorical only: {Inferred.Reason}."
        : Overridden
            ? $"Switched by you; it would be {CovariateTyping.Token(Inferred.Kind)} because {Inferred.Reason}."
            : $"{char.ToUpperInvariant(CovariateTyping.Token(Kind)[0])}{CovariateTyping.Token(Kind)[1..]} because "
              + $"{Inferred.Reason}.";

    /// <summary>Only a column of numbers can be either.</summary>
    public bool CanSwitch => Inferred.AllNumeric;

    public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;

    private void Raise(string name) =>
        PropertyChanged?.Invoke(this, new System.ComponentModel.PropertyChangedEventArgs(name));
}
