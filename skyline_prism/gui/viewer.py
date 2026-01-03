"""PRISM Viewer - Main window for exploring analysis results.

Provides an interactive viewer for PRISM output with:
- Protein/peptide document tree navigation
- Abundance plots grouped by sample type
- QC plots with configurable metadata grouping
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

# Import matplotlib with Qt backend
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class PRISMResultWidget(QWidget):
    """Result viewer widget for PRISM analysis results.

    Uses DuckDB for lazy loading - parquet files are queried on-demand
    rather than loaded entirely into memory.
    """

    def __init__(self, output_dir: Path | str, parent: QWidget | None = None) -> None:
        """Initialize the viewer widget.

        Args:
            output_dir: Path to PRISM output directory containing parquet files.
            parent: Parent widget.

        """
        super().__init__(parent)
        self.output_dir = Path(output_dir)

        # DuckDB connection for lazy loading
        import duckdb

        self.db = duckdb.connect(":memory:")

        # File paths
        self.protein_path = self.output_dir / "corrected_proteins.parquet"
        self.peptide_path = self.output_dir / "corrected_peptides.parquet"

        # Cached column names (loaded once, very fast)
        self.protein_columns: list[str] = []
        self.peptide_columns: list[str] = []

        # Mapping from protein ID to peptide list
        self.protein_peptide_map: dict[str, list[str]] = {}

        # Mapping from protein_group ID to display info (accession, description)
        self.protein_info: dict[str, dict] = {}

        # Metadata
        self.metadata: dict | None = None
        self.sample_metadata: pd.DataFrame | None = None

        # Current selection
        self.selected_protein: str | None = None
        self.selected_peptide: str | None = None

        # For compatibility with existing plot methods
        self.protein_data: pd.DataFrame | None = None
        self.peptide_data: pd.DataFrame | None = None

        self._setup_ui()
        self._init_data()
        self._populate_tree()
        # Don't auto-refresh QC plots - they're expensive

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: Tree view
        left_panel = self._create_tree_panel()
        splitter.addWidget(left_panel)

        # Right panel: Tabs
        right_panel = self._create_tabs_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([300, 900])

        # Status bar (embedded)
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)

    def _create_tree_panel(self) -> QWidget:
        """Create the left panel with search and tree."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Search box
        search_label = QLabel("Search:")
        layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter proteins/peptides...")
        self.search_box.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_box)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Proteins / Peptides"])
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree)

        return panel

    def _create_tabs_panel(self) -> QWidget:
        """Create the right panel with tabs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Abundance plot
        abundance_tab = self._create_abundance_tab()
        self.tabs.addTab(abundance_tab, "Abundance")

        # Tab 2: QC plots
        qc_tab = self._create_qc_tab()
        self.tabs.addTab(qc_tab, "QC Plots")

        return panel

    def _create_abundance_tab(self) -> QWidget:
        """Create the abundance plot tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls bar
        controls = QHBoxLayout()

        # Group by selector
        controls.addWidget(QLabel("Group by:"))
        self.group_by_combo = QComboBox()
        self.group_by_combo.addItems(["sample_type"])
        self.group_by_combo.currentTextChanged.connect(self._on_group_by_changed)
        controls.addWidget(self.group_by_combo)

        # Experimental subgroup selector
        controls.addWidget(QLabel("Subgroup experimental by:"))
        self.subgroup_combo = QComboBox()
        self.subgroup_combo.addItems(["(none)"])
        self.subgroup_combo.currentTextChanged.connect(self._on_subgroup_changed)
        controls.addWidget(self.subgroup_combo)

        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib figure
        self.abundance_figure = Figure(figsize=(10, 6), dpi=100)
        self.abundance_canvas = FigureCanvas(self.abundance_figure)
        layout.addWidget(self.abundance_canvas)

        # Navigation toolbar
        toolbar = NavigationToolbar(self.abundance_canvas, tab)
        layout.addWidget(toolbar)

        return tab

    def _init_data(self) -> None:
        """Initialize data connections and load column metadata only.

        Uses DuckDB to query parquet files on-demand rather than loading
        everything into memory.
        """
        self.status_bar.showMessage("Initializing data connections...")

        try:
            import pyarrow.parquet as pq

            # Just read schema (column names) - very fast
            if self.protein_path.exists():
                schema = pq.read_schema(self.protein_path)
                self.protein_columns = schema.names

                # Get row count efficiently
                metadata = pq.read_metadata(self.protein_path)
                row_count = metadata.num_rows
                self.status_bar.showMessage(f"Found {row_count:,} protein measurements", 3000)
            else:
                self.status_bar.showMessage("Warning: corrected_proteins.parquet not found", 5000)
                return

            if self.peptide_path.exists():
                schema = pq.read_schema(self.peptide_path)
                self.peptide_columns = schema.names
            else:
                self.status_bar.showMessage("Warning: corrected_peptides.parquet not found", 5000)

            # Load metadata JSON (small file)
            metadata_path = self.output_dir / "metadata.json"
            if metadata_path.exists():
                import json

                with open(metadata_path) as f:
                    self.metadata = json.load(f)

            # Load protein groups mapping manually (tsv file)
            # This is needed because peptide parquet doesn't have protein info
            groups_path = self.output_dir / "protein_groups.tsv"
            if groups_path.exists():
                self.status_bar.showMessage("Loading protein-peptide mapping...", 2000)
                try:
                    # Read minimal columns needed
                    groups_df = pd.read_csv(
                        groups_path, sep="\t", usecols=["GroupID", "AllPeptides"]
                    )
                    for _, row in groups_df.iterrows():
                        if pd.isna(row["AllPeptides"]):
                            continue
                        # Peptides are semicolon separated
                        peptides = [
                            p.strip() for p in str(row["AllPeptides"]).split(";") if p.strip()
                        ]
                        if peptides:
                            self.protein_peptide_map[str(row["GroupID"])] = sorted(peptides)
                except Exception as e:
                    print(f"Failed to load protein groups: {e}")

            # Load sample metadata for grouping (sample_metadata.tsv)
            sample_meta_path = self.output_dir / "sample_metadata.tsv"
            if sample_meta_path.exists():
                try:
                    self.sample_metadata = pd.read_csv(sample_meta_path, sep="\t")
                    # Ensure Replicate column exists for joining
                    if (
                        "Replicate" not in self.sample_metadata.columns
                        and "filename" in self.sample_metadata.columns
                    ):
                        self.sample_metadata["Replicate"] = self.sample_metadata["filename"]

                    self.status_bar.showMessage(
                        f"Loaded metadata for {len(self.sample_metadata)} samples", 2000
                    )
                except Exception as e:
                    print(f"Failed to load sample metadata: {e}")

            # Update group by options
            self._update_groupby_options()

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", f"Failed to initialize data: {e}")

    def _get_protein_column(self, columns: list[str]) -> str:
        """Detect the protein column name from available columns."""
        # Priority order for protein column names
        candidates = ["protein_group_id", "protein_group", "Protein", "protein"]
        for col in candidates:
            if col in columns:
                return col
        # Fallback: first column containing 'protein'
        for col in columns:
            if "protein" in col.lower():
                return col
        return columns[0] if columns else "protein"

    def _get_peptide_column(self, columns: list[str]) -> str:
        """Detect the peptide column name from available columns."""
        # Priority order for peptide column names
        candidates = ["peptide_modified", "Peptide", "peptide"]
        for col in candidates:
            if col in columns:
                return col
        # Fallback: first column containing 'peptide'
        for col in columns:
            if "peptide" in col.lower():
                return col
        return columns[0] if columns else "peptide"

    def _query_protein(self, protein_id: str) -> pd.DataFrame:
        """Query protein data for a specific protein using DuckDB.

        Handles wide-format data (samples as columns) by melting to long format.

        Args:
            protein_id: The protein group ID to query.

        Returns:
            DataFrame with abundance data for the protein in long format.

        """
        protein_col = self._get_protein_column(self.protein_columns)

        query = f"""
            SELECT * FROM read_parquet('{self.protein_path}')
            WHERE "{protein_col}" = ?
        """
        wide_df = self.db.execute(query, [protein_id]).df()

        if wide_df.empty:
            return wide_df

        # Identify metadata columns (non-sample columns)
        metadata_cols = [
            "protein_group",
            "protein_group_id",
            "leading_protein",
            "leading_name",
            "n_peptides",
            "n_unique_peptides",
            "low_confidence",
            "qc_flag",
            "Protein",
            "peptide",
            "peptide_modified",
        ]

        id_cols = [
            c
            for c in wide_df.columns
            if c.lower() in [m.lower() for m in metadata_cols] or c == protein_col
        ]
        sample_cols = [c for c in wide_df.columns if c not in id_cols]

        if not sample_cols:
            # No sample columns found - data might already be in long format
            return wide_df

        # Melt wide to long format
        long_df = wide_df.melt(
            id_vars=id_cols,
            value_vars=sample_cols,
            var_name="replicate_name",
            value_name="abundance",
        )

        return long_df

    def _query_peptide(self, peptide_id: str) -> pd.DataFrame:
        """Query peptide data for a specific peptide using DuckDB.

        Handles wide-format data (samples as columns) by melting to long format.

        Args:
            peptide_id: The peptide ID to query.

        Returns:
            DataFrame with abundance data for the peptide.

        """
        peptide_col = self._get_peptide_column(self.peptide_columns)

        query = f"""
            SELECT * FROM read_parquet('{self.peptide_path}')
            WHERE "{peptide_col}" = ?
        """
        wide_df = self.db.execute(query, [peptide_id]).df()

        if wide_df.empty:
            return wide_df

        # Identify metadata columns (non-sample columns)
        # Add peptide metadata columns
        metadata_cols = [
            "peptide_modified",
            "Peptide",
            "peptide",
            "n_transitions",
            "mean_rt",
            "Peptide Modified Sequence Unimod Ids",
        ]

        id_cols = [
            c
            for c in wide_df.columns
            if c.lower() in [m.lower() for m in metadata_cols] or c == peptide_col
        ]
        sample_cols = [c for c in wide_df.columns if c not in id_cols]

        if not sample_cols:
            return wide_df

        # Melt wide to long format
        long_df = wide_df.melt(
            id_vars=id_cols,
            value_vars=sample_cols,
            var_name="replicate_name",
            value_name="abundance",
        )

        return long_df

    def _get_unique_proteins(self) -> list[dict]:
        """Get list of unique protein info using DuckDB (fast).

        Returns:
            List of dicts with protein_group, leading_protein, leading_name,
            sorted by median abundance (descending).

        """
        if not self.protein_path.exists():
            return []

        protein_col = self._get_protein_column(self.protein_columns)

        # Get all non-metadata columns (sample columns) for median calculation
        all_cols_query = f"""
            SELECT column_name FROM (
                DESCRIBE SELECT * FROM read_parquet('{self.protein_path}')
            )
        """
        all_cols = self.db.execute(all_cols_query).fetchdf()["column_name"].tolist()

        metadata_cols = [
            "protein_group",
            "leading_protein",
            "leading_name",
            "n_peptides",
            "n_unique_peptides",
            "low_confidence",
        ]
        sample_cols = [c for c in all_cols if c not in metadata_cols]

        if sample_cols:
            # Build median calculation across all sample columns
            # Use GREATEST to handle potential nulls
            sample_expr = ", ".join([f'"{c}"' for c in sample_cols[:50]])  # Limit for performance
            query = f"""
                SELECT 
                    "{protein_col}" as protein_group,
                    COALESCE(leading_protein, "{protein_col}") as leading_protein,
                    COALESCE(leading_name, '') as leading_name,
                    MEDIAN(median_val) as median_abundance
                FROM (
                    SELECT 
                        "{protein_col}",
                        leading_protein,
                        leading_name,
                        (SELECT MEDIAN(v) FROM (VALUES ({sample_expr})) t(v)) as median_val
                    FROM read_parquet('{self.protein_path}')
                )
                GROUP BY "{protein_col}", leading_protein, leading_name
                ORDER BY median_abundance DESC NULLS LAST
            """
        else:
            query = f"""
                SELECT DISTINCT
                    "{protein_col}" as protein_group,
                    COALESCE(leading_protein, "{protein_col}") as leading_protein,
                    COALESCE(leading_name, '') as leading_name
                FROM read_parquet('{self.protein_path}')
                ORDER BY protein_group
            """

        try:
            result = self.db.execute(query).df()
        except Exception:
            # Fallback to simple query if median calc fails
            query = f"""
                SELECT DISTINCT
                    "{protein_col}" as protein_group,
                    COALESCE(leading_protein, "{protein_col}") as leading_protein,
                    COALESCE(leading_name, '') as leading_name
                FROM read_parquet('{self.protein_path}')
                ORDER BY protein_group
            """
            result = self.db.execute(query).df()

        return result.to_dict("records")

    def _get_peptides_for_protein(self, protein_id: str) -> list[str]:
        """Get list of peptides for a protein using cached map.

        Args:
            protein_id: The protein group ID.

        Returns:
            Sorted list of peptide IDs for this protein.

        """
        return self.protein_peptide_map.get(str(protein_id), [])

    def _update_groupby_options(self) -> None:
        """Update the groupby combo boxes based on available metadata columns."""
        if self.sample_metadata is None:
            return

        # Get grouping columns from metadata
        exclude_cols = {"replicate", "filename", "file_name", "sample_id", "sample", "sample_type"}

        available_cols = [
            col
            for col in self.sample_metadata.columns
            if col.lower() not in exclude_cols and self.sample_metadata[col].dtype == "object"
        ]

        # Update subgroup combo
        self.subgroup_combo.clear()
        self.subgroup_combo.addItem("(none)")
        self.subgroup_combo.addItems(available_cols)

    def _populate_tree(self) -> None:
        """Populate the tree with proteins (peptides loaded on-demand)."""
        self.tree.clear()

        if not self.protein_columns:
            return

        self.status_bar.showMessage("Loading protein list...")
        QApplication.processEvents()  # Update UI

        # Get unique proteins using DuckDB (fast even for large files)
        proteins = self._get_unique_proteins()

        if not proteins:
            self.status_bar.showMessage("No proteins found", 5000)
            return

        # Add proteins to tree (no peptides yet - load on expand)
        for pinfo in proteins:
            protein_id = pinfo.get("protein_group", "")
            accession = pinfo.get("leading_protein", protein_id)
            description = pinfo.get("leading_name", "")

            # Display format: "ACCESSION | Description" or just "ACCESSION" if no desc
            if description:
                # Truncate long descriptions
                if len(description) > 60:
                    description = description[:57] + "..."
                display_text = f"{accession} | {description}"
            else:
                display_text = accession

            protein_item = QTreeWidgetItem([display_text])
            # Store protein_group as the internal ID for queries
            protein_item.setData(0, Qt.ItemDataRole.UserRole, ("protein", protein_id))
            # Cache the protein info for use in plot titles
            self.protein_info[protein_id] = {
                "accession": accession,
                "description": description,
                "display": display_text,
            }
            # Add a dummy child so the expand arrow shows
            protein_item.addChild(QTreeWidgetItem(["Loading..."]))
            self.tree.addTopLevelItem(protein_item)

        # Connect expand signal to load peptides on-demand
        self.tree.itemExpanded.connect(self._on_item_expanded)

        self.status_bar.showMessage(f"Loaded {len(proteins)} proteins", 3000)

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        """Load peptides when a protein is expanded (lazy loading)."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return

        item_type, item_id = data

        if item_type != "protein":
            return

        # Check if already loaded (first child would not be "Loading...")
        if item.childCount() > 0:
            first_child = item.child(0)
            if first_child and first_child.text(0) != "Loading...":
                return  # Already loaded

        # Clear placeholder and load peptides
        item.takeChildren()

        self.status_bar.showMessage(f"Loading peptides for {item_id}...")
        QApplication.processEvents()

        peptides = self._get_peptides_for_protein(item_id)

        for peptide in peptides:
            peptide_item = QTreeWidgetItem([str(peptide)])
            peptide_item.setData(0, Qt.ItemDataRole.UserRole, ("peptide", peptide))
            item.addChild(peptide_item)

        self.status_bar.showMessage(f"Loaded {len(peptides)} peptides for {item_id}", 3000)

    def _on_search_changed(self, text: str) -> None:
        """Filter tree items based on search text."""
        text = text.lower()

        for i in range(self.tree.topLevelItemCount()):
            protein_item = self.tree.topLevelItem(i)
            if protein_item is None:
                continue

            protein_match = text in protein_item.text(0).lower()

            # Check peptide children
            any_peptide_match = False
            for j in range(protein_item.childCount()):
                peptide_item = protein_item.child(j)
                if peptide_item is None:
                    continue
                peptide_match = text in peptide_item.text(0).lower()
                peptide_item.setHidden(not (peptide_match or protein_match) and bool(text))
                if peptide_match:
                    any_peptide_match = True

            # Show protein if it matches or any of its peptides match
            protein_item.setHidden(not (protein_match or any_peptide_match) and bool(text))

            # Expand protein if searching and has matching peptides
            if text and any_peptide_match:
                protein_item.setExpanded(True)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle tree item selection."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return

        item_type, item_id = data

        if item_type == "protein":
            self.selected_protein = item_id
            self.selected_peptide = None
            self._plot_protein_abundance(item_id)
        elif item_type == "peptide":
            # Get parent protein
            parent = item.parent()
            if parent:
                parent_data = parent.data(0, Qt.ItemDataRole.UserRole)
                if parent_data:
                    self.selected_protein = parent_data[1]
            self.selected_peptide = item_id
            self._plot_peptide_abundance(item_id)

    def _on_group_by_changed(self, value: str) -> None:
        """Handle group by selection change."""
        self._refresh_plot()

    def _on_subgroup_changed(self, value: str) -> None:
        """Handle subgroup selection change."""
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        """Refresh the current plot with new grouping."""
        if self.selected_peptide:
            self._plot_peptide_abundance(self.selected_peptide)
        elif self.selected_protein:
            self._plot_protein_abundance(self.selected_protein)

    def _plot_protein_abundance(self, protein_id: str) -> None:
        """Plot protein abundance grouped by sample type."""
        if not self.protein_path.exists():
            return

        self.status_bar.showMessage(f"Loading data for {protein_id}...")
        QApplication.processEvents()

        # Query data for this protein using DuckDB
        plot_data = self._query_protein(protein_id)

        if plot_data.empty:
            self.status_bar.showMessage(f"No data found for {protein_id}", 3000)
            return

        # Get display name from cached protein info
        pinfo = self.protein_info.get(protein_id, {})
        display_name = pinfo.get("accession", protein_id)
        description = pinfo.get("description", "")
        if description:
            title = f"{display_name} | {description[:40]}..."
        else:
            title = f"Protein: {display_name}"
        self._create_abundance_plot(plot_data, title)
        self.status_bar.showMessage(f"Showing {len(plot_data)} samples for {protein_id}", 3000)

    def _plot_peptide_abundance(self, peptide_id: str) -> None:
        """Plot peptide abundance grouped by sample type."""
        if not self.peptide_path.exists():
            return

        self.status_bar.showMessage("Loading peptide data...")
        QApplication.processEvents()

        # Query data for this peptide using DuckDB
        plot_data = self._query_peptide(peptide_id)

        if plot_data.empty:
            self.status_bar.showMessage("No data found for peptide", 3000)
            return

        self._create_abundance_plot(plot_data, f"Peptide: {peptide_id}")

    def _create_abundance_plot(self, data: pd.DataFrame, title: str) -> None:
        """Create the abundance plot."""
        self.abundance_figure.clear()
        ax = self.abundance_figure.add_subplot(111)

        # Determine abundance column
        abundance_col = "abundance" if "abundance" in data.columns else "Area"
        if abundance_col not in data.columns:
            ax.text(0.5, 0.5, "No abundance data available", ha="center", va="center")
            self.abundance_canvas.draw()
            return

        # Merge with sample metadata if available
        if self.sample_metadata is not None:
            # Try to find common column for merge
            meta_cols = self.sample_metadata.columns
            data_cols = data.columns

            left_col = None
            right_col = None

            if "replicate_name" in data_cols:
                left_col = "replicate_name"
                if "Replicate" in meta_cols:
                    right_col = "Replicate"
                elif "filename" in meta_cols:
                    right_col = "filename"
                elif "sample_id" in meta_cols:
                    right_col = "sample_id"
                elif "replicate_name" in meta_cols:
                    right_col = "replicate_name"

            if left_col and right_col:
                # Check if values match pattern
                data = data.merge(
                    self.sample_metadata, left_on=left_col, right_on=right_col, how="left"
                )

        # Determine grouping column from the group_by combo box
        group_col = self.group_by_combo.currentText()
        if group_col not in data.columns and "sample_type" in data.columns:
            group_col = "sample_type"

        # Determine sample type column for coloring logic
        sample_type_col = "sample_type" if "sample_type" in data.columns else None

        # Determine replicate/sample column for bar plots
        sample_col = "replicate_name" if "replicate_name" in data.columns else "Replicate Name"

        # Data is now stored in linear scale in parquet files
        data = data.copy()

        # Create plot
        ax.set_ylabel("Abundance", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)

        if sample_type_col and sample_type_col in data.columns:
            # Color by sample type
            palette = {"experimental": "#1f77b4", "reference": "#2ca02c", "qc": "#ff7f0e"}

            # Get subgroup if selected
            subgroup = self.subgroup_combo.currentText()
            hue_col = sample_type_col

            if subgroup != "(none)" and subgroup in data.columns:
                # Create combined grouping for experimental samples
                data = data.copy()
                data["plot_group"] = data[sample_type_col]
                exp_mask = data[sample_type_col] == "experimental"
                data.loc[exp_mask, "plot_group"] = data.loc[exp_mask, subgroup].astype(str)
                hue_col = "plot_group"

            try:
                sns.boxplot(
                    data=data,
                    x=hue_col,
                    y=abundance_col,
                    ax=ax,
                    palette=palette if hue_col == sample_type_col else None,
                    showfliers=False,  # Hide outliers (shown as dots/circles) - users prefer swarmplot dots
                )
                sns.stripplot(
                    data=data,
                    x=hue_col,
                    y=abundance_col,
                    ax=ax,
                    color="black",
                    alpha=0.5,
                    size=3,
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Plot error: {e}", ha="center", va="center")
        else:
            # Simple bar plot by sample
            try:
                if sample_col in data.columns:
                    # Sort by sample name/id
                    plot_data = data.sort_values(sample_col)

                    # Use a faster plotting method for many bars
                    x_pos = np.arange(len(plot_data))
                    ax.bar(x_pos, plot_data[abundance_col])

                    # Hide x-axis labels if too many
                    if len(plot_data) > 50:
                        ax.set_xticks([])
                        ax.set_xlabel("Replicates (sorted)")
                    else:
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(plot_data[sample_col], rotation=45, ha="right")

            except Exception as e:
                ax.text(0.5, 0.5, f"Plot error: {e}", ha="center", va="center")

        ax.set_title(title, fontsize=14)
        # ax.set_xlabel("") # Handled above

        self.abundance_figure.tight_layout()
        self.abundance_canvas.draw()

    def _create_qc_tab(self) -> QWidget:
        """Create the QC plots tab with on-demand data loading."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Add Load Data button at top
        controls = QHBoxLayout()
        self.load_qc_btn = QPushButton("Load QC Data (may take a moment)")
        self.load_qc_btn.clicked.connect(self._load_qc_data)
        controls.addWidget(self.load_qc_btn)
        controls.addStretch()
        layout.addLayout(controls)

        # Create subtabs for different QC plots
        self.qc_subtabs = QTabWidget()
        layout.addWidget(self.qc_subtabs)

        # PCA Plot subtab
        pca_tab = self._create_pca_subtab()
        self.qc_subtabs.addTab(pca_tab, "PCA")

        # CV Distribution subtab
        cv_tab = self._create_cv_subtab()
        self.qc_subtabs.addTab(cv_tab, "CV Distribution")

        # Correlation Heatmap subtab - temporarily disabled, will be added back later
        # corr_tab = self._create_correlation_subtab()
        # self.qc_subtabs.addTab(corr_tab, "Correlation")

        # Intensity Distribution subtab
        intensity_tab = self._create_intensity_subtab()
        self.qc_subtabs.addTab(intensity_tab, "Intensity")

        return tab

    def _load_qc_data(self) -> None:
        """Load full protein data for QC plots (on-demand)."""
        if not self.protein_path.exists():
            QMessageBox.warning(self, "No Data", "Protein data file not found")
            return

        self.status_bar.showMessage("Loading protein data for QC plots...")
        self.load_qc_btn.setEnabled(False)
        self.load_qc_btn.setText("Loading...")
        QApplication.processEvents()

        try:
            # Load full protein data for QC analysis
            # By default read_parquet returns wide format if that's how it's saved
            df = pd.read_parquet(self.protein_path)

            # Check if wide format (abundance columns as cols)
            # Identify metadata columns
            known_meta = {
                "protein_group",
                "protein_group_id",
                "leading_protein",
                "leading_name",
                "n_peptides",
                "n_unique_peptides",
                "low_confidence",
                "mean_rt",
            }
            meta_cols = [c for c in df.columns if c in known_meta]

            # If we have sample columns as columns, we need to melt
            # Assume anything not known metadata is a sample if it looks like one
            if "abundance" not in df.columns:
                # It's likely wide format. wide columns = everything else
                id_vars = meta_cols
                value_vars = [c for c in df.columns if c not in id_vars]

                self.protein_data = df.melt(
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name="replicate_name",
                    value_name="abundance",
                )
            else:
                self.protein_data = df

            # Merge with metadata if available
            if self.sample_metadata is not None:
                # Try to find common column for merge
                meta_cols_df = self.sample_metadata.columns
                data_cols = self.protein_data.columns

                left_col = None
                right_col = None

                if "replicate_name" in data_cols:
                    left_col = "replicate_name"
                    if "Replicate" in meta_cols_df:
                        right_col = "Replicate"
                    elif "filename" in meta_cols_df:
                        right_col = "filename"
                    elif "sample_id" in meta_cols_df:
                        right_col = "sample_id"
                    elif "replicate_name" in meta_cols_df:
                        right_col = "replicate_name"

                if left_col and right_col:
                    self.protein_data = self.protein_data.merge(
                        self.sample_metadata, left_on=left_col, right_on=right_col, how="left"
                    )

            self.status_bar.showMessage(f"Loaded {len(self.protein_data):,} protein measurements")
            self.load_qc_btn.setText("Data Loaded ✓")

            # Now refresh the plots
            self._refresh_qc_plots()

        except Exception as e:
            self.load_qc_btn.setEnabled(True)
            self.load_qc_btn.setText("Load QC Data (retry)")
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

    def _create_pca_subtab(self) -> QWidget:
        """Create the PCA plot subtab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Color by:"))
        self.pca_colorby = QComboBox()
        self.pca_colorby.addItems(["sample_type", "batch"])
        self.pca_colorby.currentTextChanged.connect(self._plot_pca)
        controls.addWidget(self.pca_colorby)
        controls.addStretch()
        layout.addLayout(controls)

        # Figure
        self.pca_figure = Figure(figsize=(10, 6), dpi=100)
        self.pca_canvas = FigureCanvas(self.pca_figure)
        layout.addWidget(self.pca_canvas)

        toolbar = NavigationToolbar(self.pca_canvas, tab)
        layout.addWidget(toolbar)

        return tab

    def _create_cv_subtab(self) -> QWidget:
        """Create the CV distribution subtab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Figure
        self.cv_figure = Figure(figsize=(10, 6), dpi=100)
        self.cv_canvas = FigureCanvas(self.cv_figure)
        layout.addWidget(self.cv_canvas)

        toolbar = NavigationToolbar(self.cv_canvas, tab)
        layout.addWidget(toolbar)

        return tab

    def _create_correlation_subtab(self) -> QWidget:
        """Create the correlation heatmap subtab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Sample type:"))
        self.corr_sample_type = QComboBox()
        self.corr_sample_type.addItems(["reference", "qc", "all"])
        self.corr_sample_type.currentTextChanged.connect(self._plot_correlation)
        controls.addWidget(self.corr_sample_type)
        controls.addStretch()
        layout.addLayout(controls)

        # Figure
        self.corr_figure = Figure(figsize=(10, 8), dpi=100)
        self.corr_canvas = FigureCanvas(self.corr_figure)
        layout.addWidget(self.corr_canvas)

        toolbar = NavigationToolbar(self.corr_canvas, tab)
        layout.addWidget(toolbar)

        return tab

    def _create_intensity_subtab(self) -> QWidget:
        """Create the intensity distribution subtab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Figure
        self.intensity_figure = Figure(figsize=(10, 6), dpi=100)
        self.intensity_canvas = FigureCanvas(self.intensity_figure)
        layout.addWidget(self.intensity_canvas)

        toolbar = NavigationToolbar(self.intensity_canvas, tab)
        layout.addWidget(toolbar)

        return tab

    def _plot_pca(self, colorby: str | None = None) -> None:
        """Plot PCA of protein abundances."""
        if self.protein_data is None:
            return

        self.pca_figure.clear()
        ax = self.pca_figure.add_subplot(111)

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Pivot data for PCA: samples as rows, proteins as columns
            abundance_col = "abundance" if "abundance" in self.protein_data.columns else "Area"

            sample_col = None
            for col in ["replicate_name", "Replicate Name", "sample_id", "filename"]:
                if col in self.protein_data.columns:
                    sample_col = col
                    break

            protein_col = None
            for col in ["protein_group", "protein_group_id", "leading_protein", "Protein"]:
                if col in self.protein_data.columns:
                    protein_col = col
                    break

            if not abundance_col or not sample_col or not protein_col:
                ax.text(0.5, 0.5, "Missing required columns for PCA", ha="center", va="center")
                self.pca_canvas.draw()
                return

            if abundance_col not in self.protein_data.columns:
                ax.text(0.5, 0.5, "No abundance data for PCA", ha="center", va="center")
                self.pca_canvas.draw()
                return

            pivot = self.protein_data.pivot_table(
                index=sample_col, columns=protein_col, values=abundance_col, aggfunc="mean"
            ).dropna(axis=1)

            if pivot.shape[1] < 2:
                ax.text(0.5, 0.5, "Insufficient data for PCA", ha="center", va="center")
                self.pca_canvas.draw()
                return

            # Get sample metadata for coloring
            colorby = colorby or self.pca_colorby.currentText()
            sample_type_col = "sample_type" if "sample_type" in self.protein_data.columns else None

            # Standardize and run PCA
            # Convert from linear to log2 for variance stabilization (PCA works better on log scale)
            # Output parquet files are LINEAR, but PCA should use LOG2
            log2_data = np.log2(pivot.values + 1)  # +1 to avoid log(0)
            log2_data = np.nan_to_num(log2_data, nan=0, posinf=0, neginf=0)
            scaled_data = StandardScaler().fit_transform(log2_data)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(scaled_data)

            # Create color mapping
            if sample_type_col and sample_type_col in self.protein_data.columns:
                sample_to_type = (
                    self.protein_data.drop_duplicates(sample_col)
                    .set_index(sample_col)[sample_type_col]
                    .to_dict()
                )
                colors = [sample_to_type.get(s, "unknown") for s in pivot.index]
                palette = {
                    "experimental": "#1f77b4",
                    "reference": "#2ca02c",
                    "qc": "#ff7f0e",
                    "unknown": "#7f7f7f",
                }

                for sample_type, color in palette.items():
                    mask = np.array([c == sample_type for c in colors])
                    if mask.any():
                        ax.scatter(
                            coords[mask, 0],
                            coords[mask, 1],
                            c=color,
                            label=sample_type,
                            alpha=0.7,
                            s=50,
                        )

                ax.legend()
            else:
                ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=50)

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
            ax.set_title("PCA of Protein Abundances")

        except Exception as e:
            ax.text(0.5, 0.5, f"PCA error: {e}", ha="center", va="center", fontsize=10)

        self.pca_figure.tight_layout()
        self.pca_canvas.draw()

    def _plot_cv_distribution(self) -> None:
        """Plot coefficient of variation distribution."""
        if self.protein_data is None:
            return

        self.cv_figure.clear()
        ax = self.cv_figure.add_subplot(111)

        try:
            abundance_col = "abundance" if "abundance" in self.protein_data.columns else "Area"

            protein_col = None
            for col in ["protein_group", "protein_group_id", "leading_protein", "Protein"]:
                if col in self.protein_data.columns:
                    protein_col = col
                    break

            sample_type_col = "sample_type" if "sample_type" in self.protein_data.columns else None

            if not abundance_col or not protein_col:
                ax.text(0.5, 0.5, "Missing columns for CV analysis", ha="center", va="center")
                self.cv_canvas.draw()
                return

            if abundance_col not in self.protein_data.columns:
                ax.text(0.5, 0.5, "No abundance data for CV", ha="center", va="center")
                self.cv_canvas.draw()
                return

            # Calculate CV per protein per sample type
            if sample_type_col and sample_type_col in self.protein_data.columns:
                cv_data = []
                for stype in self.protein_data[sample_type_col].unique():
                    subset = self.protein_data[self.protein_data[sample_type_col] == stype]
                    cv_by_protein = subset.groupby(protein_col)[abundance_col].agg(["mean", "std"])
                    cv_by_protein["cv"] = cv_by_protein["std"] / cv_by_protein["mean"] * 100
                    cv_by_protein = cv_by_protein.dropna()
                    if not cv_by_protein.empty:
                        cv_data.append((stype, cv_by_protein["cv"].values))

                colors = {"experimental": "#1f77b4", "reference": "#2ca02c", "qc": "#ff7f0e"}
                for stype, cvs in cv_data:
                    color = colors.get(stype, "#7f7f7f")
                    ax.hist(
                        cvs,
                        bins=50,
                        alpha=0.5,
                        label=f"{stype} (median: {np.median(cvs):.1f}%)",
                        color=color,
                    )

                ax.legend()
            else:
                cv_by_protein = self.protein_data.groupby(protein_col)[abundance_col].agg(
                    ["mean", "std"]
                )
                cv_by_protein["cv"] = cv_by_protein["std"] / cv_by_protein["mean"] * 100
                cv_by_protein = cv_by_protein.dropna()
                ax.hist(cv_by_protein["cv"].values, bins=50, alpha=0.7)

            ax.set_xlabel("Coefficient of Variation (%)")
            ax.set_ylabel("Count")
            ax.set_title("CV Distribution by Sample Type")
            ax.set_xlim(0, 100)  # Limit to 0-100% like qc-report.html
            ax.axvline(x=20, color="red", linestyle="--", alpha=0.5, label="20% threshold")

        except Exception as e:
            ax.text(0.5, 0.5, f"CV plot error: {e}", ha="center", va="center", fontsize=10)

        self.cv_figure.tight_layout()
        self.cv_canvas.draw()

    def _plot_correlation(self, sample_type: str | None = None) -> None:
        """Plot sample correlation heatmap."""
        if self.protein_data is None:
            return

        self.corr_figure.clear()
        ax = self.corr_figure.add_subplot(111)

        try:
            abundance_col = "abundance" if "abundance" in self.protein_data.columns else "Area"

            sample_col = None
            for col in ["replicate_name", "Replicate Name", "sample_id", "filename"]:
                if col in self.protein_data.columns:
                    sample_col = col
                    break

            protein_col = None
            for col in ["protein_group", "protein_group_id", "leading_protein", "Protein"]:
                if col in self.protein_data.columns:
                    protein_col = col
                    break

            sample_type_col = "sample_type" if "sample_type" in self.protein_data.columns else None

            sample_type = sample_type or self.corr_sample_type.currentText()

            if not sample_col or not protein_col:
                ax.text(0.5, 0.5, "Missing columns for correlation", ha="center", va="center")
                self.corr_canvas.draw()
                return

            # Filter by sample type if specified
            data = self.protein_data.copy()
            if sample_type != "all" and sample_type_col and sample_type_col in data.columns:
                data = data[data[sample_type_col] == sample_type]

            if data.empty:
                ax.text(0.5, 0.5, f"No {sample_type} samples found", ha="center", va="center")
                self.corr_canvas.draw()
                return

            # Pivot for correlation
            pivot = data.pivot_table(
                index=protein_col, columns=sample_col, values=abundance_col, aggfunc="mean"
            ).dropna()

            if pivot.shape[1] < 2:
                ax.text(
                    0.5, 0.5, "Need at least 2 samples for correlation", ha="center", va="center"
                )
                self.corr_canvas.draw()
                return

            corr = pivot.corr()

            # Plot heatmap
            sns.heatmap(
                corr,
                ax=ax,
                cmap="RdYlGn",
                vmin=0.8,
                vmax=1.0,
                annot=corr.shape[0] <= 10,
                fmt=".2f",
                square=True,
            )
            ax.set_title(f"Sample Correlation ({sample_type})")

        except Exception as e:
            ax.text(0.5, 0.5, f"Correlation error: {e}", ha="center", va="center", fontsize=10)

        self.corr_figure.tight_layout()
        self.corr_canvas.draw()

    def _plot_intensity_distribution(self) -> None:
        """Plot intensity distribution."""
        if self.protein_data is None:
            return

        self.intensity_figure.clear()
        ax = self.intensity_figure.add_subplot(111)

        try:
            abundance_col = "abundance" if "abundance" in self.protein_data.columns else "Area"
            sample_type_col = "sample_type" if "sample_type" in self.protein_data.columns else None

            if abundance_col not in self.protein_data.columns:
                ax.text(0.5, 0.5, "No abundance data", ha="center", va="center")
                self.intensity_canvas.draw()
                return

            values = self.protein_data[abundance_col].dropna()
            log_values = np.log2(values[values > 0])

            if sample_type_col and sample_type_col in self.protein_data.columns:
                colors = {"experimental": "#1f77b4", "reference": "#2ca02c", "qc": "#ff7f0e"}
                for stype in self.protein_data[sample_type_col].unique():
                    subset = self.protein_data[self.protein_data[sample_type_col] == stype][
                        abundance_col
                    ]
                    subset = subset[subset > 0]
                    if not subset.empty:
                        color = colors.get(stype, "#7f7f7f")
                        ax.hist(np.log2(subset), bins=50, alpha=0.5, label=stype, color=color)
                ax.legend()
            else:
                ax.hist(log_values, bins=50, alpha=0.7)

            ax.set_xlabel("log2(Abundance)")
            ax.set_ylabel("Count")
            ax.set_title("Intensity Distribution")

        except Exception as e:
            ax.text(0.5, 0.5, f"Intensity plot error: {e}", ha="center", va="center", fontsize=10)

        self.intensity_figure.tight_layout()
        self.intensity_canvas.draw()

    def _refresh_qc_plots(self) -> None:
        """Refresh all QC plots after data is loaded."""
        self._plot_pca()
        self._plot_cv_distribution()
        # self._plot_correlation()  # Temporarily disabled
        self._plot_intensity_distribution()


class PRISMViewer(QMainWindow):
    """Main viewer window wrapper for PRISM analysis results (standalone)."""

    def __init__(self, output_dir: Path | str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PRISM Viewer")
        self.setMinimumSize(1200, 800)

        self.result_widget = PRISMResultWidget(output_dir)
        self.setCentralWidget(self.result_widget)


def main() -> int:
    """Launch the PRISM viewer."""
    if len(sys.argv) < 2:
        print("Usage: python -m skyline_prism.gui.viewer <output_dir>")
        return 1

    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        return 1

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    viewer = PRISMViewer(output_dir)
    viewer.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
