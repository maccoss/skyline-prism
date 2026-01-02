"""PRISM Configuration Widget for the Unified GUI.

Provides a PyQt6-based widget for allowing users to select input files
and configure pipeline parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)


class PRISMConfigWidget(QWidget):
    """Configuration widget for PRISM pipeline parameters.

    Includes input file selection and algorithm settings.
    """

    # Signal emitted when "Run Analysis" is clicked with valid config
    run_analysis_requested = pyqtSignal(dict)

    def __init__(
        self,
        has_batch_column: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.has_batch_column = has_batch_column
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        main_layout = QVBoxLayout(self)

        # Scroll area for configurations if window is small
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(content_widget)

        # Input Files Section
        layout.addWidget(self._create_input_files_group())

        # Transition Rollup section
        layout.addWidget(self._create_transition_rollup_group())

        # Peptide Normalization section
        layout.addWidget(self._create_peptide_normalization_group())

        # Protein Inference section
        layout.addWidget(self._create_protein_inference_group())

        # Protein Rollup section
        layout.addWidget(self._create_protein_rollup_group())

        # Protein Normalization section
        layout.addWidget(self._create_protein_normalization_group())

        # Add stretch at the bottom
        layout.addStretch()

        main_layout.addWidget(scroll)

        # Run Button Area
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setMinimumWidth(150)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self._on_run_clicked)
        btn_layout.addWidget(self.run_btn)

        main_layout.addLayout(btn_layout)

    def _create_input_files_group(self) -> QGroupBox:
        """Create the Input Files configuration group."""
        group = QGroupBox("Input Files")
        layout = QFormLayout(group)

        # Report Files
        self.report_path = QLineEdit()
        self.report_path.setPlaceholderText("Select one or more Skyline CSV reports...")
        btn_report = QPushButton("Browse...")
        btn_report.clicked.connect(self._browse_report)

        h_report = QHBoxLayout()
        h_report.addWidget(self.report_path)
        h_report.addWidget(btn_report)
        layout.addRow("Skyline Report(s):", h_report)

        # Metadata File(s)
        self.metadata_path = QLineEdit()
        self.metadata_path.setPlaceholderText("Select one or more Metadata/Replicates CSVs...")
        btn_meta = QPushButton("Browse...")
        btn_meta.clicked.connect(self._browse_metadata)

        h_meta = QHBoxLayout()
        h_meta.addWidget(self.metadata_path)
        h_meta.addWidget(btn_meta)
        layout.addRow("Metadata File(s):", h_meta)

        # Output Directory
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Select output directory...")
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_output)

        h_out = QHBoxLayout()
        h_out.addWidget(self.output_dir)
        h_out.addWidget(btn_out)
        layout.addRow("Output Directory:", h_out)

        return group

    def _create_transition_rollup_group(self) -> QGroupBox:
        """Create the Transition Rollup configuration group."""
        group = QGroupBox("Transition Rollup")
        layout = QFormLayout(group)

        # Method dropdown
        self.transition_method = QComboBox()
        self.transition_method.addItems(["adaptive", "median_polish", "sum", "topn"])
        self.transition_method.setCurrentText("adaptive")
        self.transition_method.currentTextChanged.connect(self._on_transition_method_changed)
        layout.addRow("Method:", self.transition_method)

        # Min transitions
        self.min_transitions = QSpinBox()
        self.min_transitions.setRange(1, 10)
        self.min_transitions.setValue(3)
        layout.addRow("Min Transitions:", self.min_transitions)

        # Learn weights checkbox (only for adaptive)
        self.learn_weights = QCheckBox("Learn weights from reference samples")
        self.learn_weights.setChecked(True)
        layout.addRow("", self.learn_weights)

        return group

    def _create_peptide_normalization_group(self) -> QGroupBox:
        """Create the Peptide Normalization configuration group."""
        group = QGroupBox("Peptide Normalization")
        layout = QFormLayout(group)

        # Method dropdown
        self.peptide_norm_method = QComboBox()
        self.peptide_norm_method.addItems(["rt_lowess", "median", "none"])
        self.peptide_norm_method.setCurrentText("rt_lowess")
        layout.addRow("Method:", self.peptide_norm_method)

        # Batch correction checkbox
        self.peptide_batch_correction = QCheckBox("Batch Correction")
        self.peptide_batch_correction.setChecked(self.has_batch_column)
        if not self.has_batch_column:
            self.peptide_batch_correction.setEnabled(False)
            self.peptide_batch_correction.setToolTip(
                "Disabled: No 'Batch' column found in metadata"
            )
        layout.addRow("", self.peptide_batch_correction)

        return group

    def _create_protein_inference_group(self) -> QGroupBox:
        """Create the Protein Inference configuration group."""
        group = QGroupBox("Protein Inference")
        layout = QFormLayout(group)

        # FASTA file picker
        fasta_layout = QHBoxLayout()
        self.fasta_path = QLineEdit()
        self.fasta_path.setPlaceholderText("(Optional) Select FASTA file...")
        fasta_layout.addWidget(self.fasta_path)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_fasta)
        fasta_layout.addWidget(browse_btn)

        layout.addRow("FASTA File:", fasta_layout)

        # Shared peptide handling
        self.shared_peptides = QComboBox()
        self.shared_peptides.addItems(["all_groups", "unique_only", "razor"])
        self.shared_peptides.setCurrentText("all_groups")
        layout.addRow("Shared Peptides:", self.shared_peptides)

        return group

    def _create_protein_rollup_group(self) -> QGroupBox:
        """Create the Protein Rollup configuration group."""
        group = QGroupBox("Protein Rollup")
        layout = QFormLayout(group)

        # Method dropdown
        self.protein_rollup_method = QComboBox()
        self.protein_rollup_method.addItems(
            [
                "median_polish",
                "sum",
                "topn",
                "ibaq (experimental)",
                "maxlfq (experimental)",
            ]
        )
        self.protein_rollup_method.setCurrentText("median_polish")
        layout.addRow("Method:", self.protein_rollup_method)

        # Min peptides
        self.min_peptides = QSpinBox()
        self.min_peptides.setRange(1, 10)
        self.min_peptides.setValue(2)
        layout.addRow("Min Peptides:", self.min_peptides)

        return group

    def _create_protein_normalization_group(self) -> QGroupBox:
        """Create the Protein Normalization configuration group."""
        group = QGroupBox("Protein Normalization")
        layout = QFormLayout(group)

        # Method dropdown
        self.protein_norm_method = QComboBox()
        self.protein_norm_method.addItems(["median", "none"])
        self.protein_norm_method.setCurrentText("median")
        layout.addRow("Method:", self.protein_norm_method)

        # Batch correction checkbox
        self.protein_batch_correction = QCheckBox("Batch Correction")
        self.protein_batch_correction.setChecked(self.has_batch_column)
        if not self.has_batch_column:
            self.protein_batch_correction.setEnabled(False)
            self.protein_batch_correction.setToolTip(
                "Disabled: No 'Batch' column found in metadata"
            )
        layout.addRow("", self.protein_batch_correction)

        return group

    def _on_transition_method_changed(self, method: str) -> None:
        """Handle transition method change."""
        self.learn_weights.setEnabled(method == "adaptive")
        if method != "adaptive":
            self.learn_weights.setChecked(False)

    def _browse_report(self) -> None:
        """Open file dialog to select Report CSVs."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Skyline Report CSVs",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if files:
            # We can support glob patterns or just first file for now in the line edit,
            # ideally we join them or show count.
            # For this MVP let's join with semicolon or just pick first + glob if multiple?
            # User uses Prism run with a glob usually.
            # If multiple files selected, we can try to find a common glob or list them.
            # Simple approach: join with ;
            self.report_path.setText(";".join(files))

    def _browse_metadata(self) -> None:
        """Open file dialog to select multiple Metadata CSVs."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Metadata CSVs",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if files:
            self.metadata_path.setText(";".join(files))

    def _browse_output(self) -> None:
        """Open file dialog to select Output Directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
        )
        if dir_path:
            self.output_dir.setText(dir_path)

    def _browse_fasta(self) -> None:
        """Open file dialog to select FASTA file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FASTA File",
            "",
            "FASTA Files (*.fasta *.fa *.faa);;All Files (*)",
        )
        if file_path:
            self.fasta_path.setText(file_path)

    def _on_run_clicked(self) -> None:
        """Handle Run button click."""
        # Validation
        if not self.report_path.text():
            self._show_error("Please select at least one Skyline Report CSV.")
            return
        if not self.metadata_path.text():
            self._show_error("Please select a Metadata CSV.")
            return
        if not self.output_dir.text():
            self._show_error("Please select an Output Directory.")
            return

        method = self.protein_rollup_method.currentText()
        if "ibaq" in method or "maxlfq" in method:
            if not self.fasta_path.text():
                self._show_error(f"The {method.split()[0]} method requires a FASTA file.")
                return

        self._save_settings()
        self.run_analysis_requested.emit(self.get_config())

    def _show_error(self, message: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.warning(self, "Validation Error", message)

    def set_inputs(self, report: str, metadata: str, output: str) -> None:
        """Pre-fill input fields (e.g. when called from Skyline)."""
        if report:
            self.report_path.setText(report)
        if metadata:
            self.metadata_path.setText(metadata)
        if output:
            self.output_dir.setText(output)

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary from widget values."""
        protein_method = self.protein_rollup_method.currentText()
        if "(" in protein_method:
            protein_method = protein_method.split()[0]

        config: dict[str, Any] = {
            "inputs": {
                "report": self.report_path.text(),
                "metadata": self.metadata_path.text(),
                "output": self.output_dir.text(),
            },
            "transition_rollup": {
                "enabled": True,
                "method": self.transition_method.currentText(),
                "min_transitions": self.min_transitions.value(),
                "learn_adaptive_weights": self.learn_weights.isChecked(),
            },
            "global_normalization": {
                "method": self.peptide_norm_method.currentText(),
            },
            "peptide_batch_correction": {
                "enabled": self.peptide_batch_correction.isChecked() and self.has_batch_column,
            },
            "protein_rollup": {
                "method": protein_method,
                "min_peptides": self.min_peptides.value(),
            },
            "protein_normalization": {
                "method": self.protein_norm_method.currentText(),
            },
            "protein_batch_correction": {
                "enabled": self.protein_batch_correction.isChecked() and self.has_batch_column,
            },
        }

        # Add parsimony if FASTA provided
        fasta_path = self.fasta_path.text()
        if fasta_path:
            config["parsimony"] = {
                "enabled": True,
                "fasta_path": fasta_path,
                "shared_peptide_handling": self.shared_peptides.currentText(),
            }
        else:
            config["parsimony"] = {
                "enabled": False,
            }

        return config

    def _get_settings_path(self) -> Path:
        """Get the path to the settings file."""
        config_dir = Path.home() / ".config" / "skyline-prism"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "gui_settings.json"

    def _save_settings(self) -> None:
        """Save current settings for next use."""
        settings = {
            # We don't save input paths usually, or only last output dir?
            # Let's save output directory at least
            "last_output_dir": self.output_dir.text(),
            "transition_method": self.transition_method.currentText(),
            "min_transitions": self.min_transitions.value(),
            "learn_weights": self.learn_weights.isChecked(),
            "peptide_norm_method": self.peptide_norm_method.currentText(),
            "peptide_batch_correction": self.peptide_batch_correction.isChecked(),
            "fasta_path": self.fasta_path.text(),
            "shared_peptides": self.shared_peptides.currentText(),
            "protein_rollup_method": self.protein_rollup_method.currentText(),
            "min_peptides": self.min_peptides.value(),
            "protein_norm_method": self.protein_norm_method.currentText(),
            "protein_batch_correction": self.protein_batch_correction.isChecked(),
        }

        try:
            with open(self._get_settings_path(), "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _load_settings(self) -> None:
        """Load previously saved settings."""
        settings_path = self._get_settings_path()
        if not settings_path.exists():
            return

        try:
            with open(settings_path) as f:
                settings = json.load(f)

            if "last_output_dir" in settings and not self.output_dir.text():
                # Only load if empty (not pre-filled by Skyline)
                self.output_dir.setText(settings["last_output_dir"])

            if "transition_method" in settings:
                self.transition_method.setCurrentText(settings["transition_method"])
            if "min_transitions" in settings:
                self.min_transitions.setValue(settings["min_transitions"])
            if "learn_weights" in settings:
                self.learn_weights.setChecked(settings["learn_weights"])
            if "peptide_norm_method" in settings:
                self.peptide_norm_method.setCurrentText(settings["peptide_norm_method"])
            if "peptide_batch_correction" in settings and self.has_batch_column:
                self.peptide_batch_correction.setChecked(settings["peptide_batch_correction"])
            if "fasta_path" in settings:
                self.fasta_path.setText(settings["fasta_path"])
            if "shared_peptides" in settings:
                self.shared_peptides.setCurrentText(settings["shared_peptides"])
            if "protein_rollup_method" in settings:
                self.protein_rollup_method.setCurrentText(settings["protein_rollup_method"])
            if "min_peptides" in settings:
                self.min_peptides.setValue(settings["min_peptides"])
            if "protein_norm_method" in settings:
                self.protein_norm_method.setCurrentText(settings["protein_norm_method"])
            if "protein_batch_correction" in settings and self.has_batch_column:
                self.protein_batch_correction.setChecked(settings["protein_batch_correction"])
        except Exception:
            pass
