#nullable enable

using System;
using System.Collections.Generic;

namespace SkylinePrism.Skyline;

/// <summary>Inline rows read from a Skyline report/document-grid view (column headers + string cells).</summary>
public sealed record ReportRows(IReadOnlyList<string> Columns, IReadOnlyList<string[]> Rows);

/// <summary>
/// The subset of Skyline's JSON-RPC client that <see cref="SkylineReportDriver"/> needs. Extracting
/// it as an interface lets the driver be tested with a fake instead of a live Skyline pipe.
/// </summary>
public interface ISkylineClient
{
    string GetDocumentPath();
    string GetVersion();
    void ExportReport(string reportName, string filePath, string culture);
    string[] GetSettingsListNames(string listType, string? groupName);

    /// <summary>Names of the settings-list items currently active in the document (e.g. the selected enzyme).</summary>
    string[] GetSettingsListSelectedItems(string listType);

    /// <summary>XML definition of a single settings-list item (e.g. an enzyme's cut/no_cut/sense).</summary>
    string GetSettingsListItem(string listType, string itemName);

    void RunCommandSilent(string[] args);

    /// <summary>Column names of the Replicate entity (built-ins + user-defined document annotations).</summary>
    string[] GetReplicateColumns();

    /// <summary>Read a per-replicate report (one row per replicate) for the given columns; null if unavailable.</summary>
    ReportRows? GetReplicateReport(IReadOnlyList<string> selectColumns);

    /// <summary>
    /// Select an element in Skyline's document tree by ElementLocator, e.g.
    /// <c>MoleculeGroup:/sp|P02768|ALBU_HUMAN</c> or <c>Molecule:/&lt;protein&gt;/&lt;peptide&gt;</c>.
    /// This is what makes a plot point clickable: picking a protein here navigates the user's document.
    /// </summary>
    void SetSelectedElement(string elementLocator);

    /// <summary>
    /// The locator of what is selected in Skyline right now, at the requested tree level, or null when
    /// nothing of that level is selected.
    /// <para>
    /// <paramref name="elementType"/> is <c>"MoleculeGroup"</c> (protein), <c>"Molecule"</c> (peptide),
    /// <c>"Precursor"</c> or <c>"Transition"</c>. Skyline resolves the ANCESTOR at that level, so asking
    /// for the protein while a transition is selected returns the protein containing it - which is what
    /// lets a plot follow the selection without parsing locator strings, and works the same whether the
    /// user clicked the Targets tree or a row in a document grid.
    /// </para>
    /// </summary>
    string? GetSelectedElementLocator(string elementType);

    /// <summary>
    /// Every element at a tree level ("group" = proteins, "molecule" = peptides) as (name, locator).
    /// Read once and cached, so a click maps a plotted protein to the locator Skyline itself uses rather
    /// than one built by string surgery.
    /// </summary>
    IReadOnlyList<(string Name, string Locator)> GetLocations(string level);
}

/// <summary>
/// Opens a Skyline connection per call and hands the driver an <see cref="ISkylineClient"/>.
/// <see cref="SkylineSession"/> is the production implementation (a JSON-RPC pipe); tests supply a fake.
/// </summary>
public interface ISkylineExecutor
{
    T Execute<T>(Func<ISkylineClient, T> action);
    void Execute(Action<ISkylineClient> action);
}
