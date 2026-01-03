"""PRISM Main Window for the Unified GUI.

Combines Configuration, Console, and Results into a single window.
"""

from __future__ import annotations

import sys
from typing import Any

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from skyline_prism.gui.config_dialog import PRISMConfigWidget
from skyline_prism.gui.console_widget import PRISMConsoleWidget
from skyline_prism.gui.viewer import PRISMResultWidget


class PRISMMainWindow(QMainWindow):
    """Unified access point for PRISM: Config -> Run -> View."""

    def __init__(
        self,
        report_file: str = "",
        metadata_file: str = "",
        output_dir: str = "",
        has_batch_column: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Skyline-PRISM")
        self.resize(1200, 800)

        # Initial args from CLI
        self._initial_report = report_file
        self._initial_metadata = metadata_file
        self._initial_output = output_dir
        self._has_batch_column = has_batch_column

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the tabbed interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Configuration
        self.config_tab = PRISMConfigWidget(has_batch_column=self._has_batch_column)
        # Pre-fill inputs if provided
        self.config_tab.set_inputs(
            self._initial_report, self._initial_metadata, self._initial_output
        )
        self.config_tab.run_analysis_requested.connect(self._on_run_requested)
        self.tabs.addTab(self.config_tab, "Configuration")

        # Tab 2: Processing (Console)
        self.console_tab = PRISMConsoleWidget()
        self.console_tab.process_finished.connect(self._on_process_finished)
        self.tabs.addTab(self.console_tab, "Processing")

        # Tab 3: Results (Viewer) - initially placeholder or empty
        # We will create/replace this tab upon successful run
        self.result_tab = QWidget()  # Placeholder
        self.tabs.addTab(self.result_tab, "Results")
        self.tabs.setTabEnabled(2, False)

    def _on_run_requested(self, config: dict[str, Any]) -> None:
        """Handle Run Analysis request from Config tab."""
        # 1. Generate command arguments
        inputs = config.get("inputs", {})
        report_path = inputs.get("report", "")
        metadata_path = inputs.get("metadata", "")
        output_dir = inputs.get("output", "")

        if not report_path or not metadata_path or not output_dir:
            QMessageBox.warning(self, "Error", "Missing input files.")
            return

        # Prepare arguments for 'prism run'
        # We need to construct the CLI command.
        # Since we are running from source or installed package, we use 'prism' or sys.executable + -m skyline_prism

        args = ["run"]

        # Handle multiple reports if separated by semicolon
        reports = report_path.split(";")
        for r in reports:
            if r.strip():
                args.extend(["--input", r.strip()])

        # Handle multiple metadata files if separated by semicolon
        metadata_list = metadata_path.split(";")
        for m in metadata_list:
            if m.strip():
                args.extend(["--metadata", m.strip()])

        args.extend(["--output", output_dir])

        # Add config file (we need to dump the config dict to a temp yaml file first)
        import tempfile

        import yaml

        # Create a temporary config file
        # We shouldn't delete it immediately as the process needs to read it.
        # Let's save it in the output dir (created by us if needed, or by run command)
        # Actually, prism run creates output dir.
        # Safe bet: system temp, or user config dir.
        # Let's use a temp file that persists logic until process start?
        # QProcess is async. We need a file that exists.

        try:
            tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
            # Remove 'inputs' key as it's not part of yaml config for logic
            run_config = config.copy()
            if "inputs" in run_config:
                del run_config["inputs"]

            yaml.dump(run_config, tmp_config)
            tmp_config.close()

            args.extend(["--config", tmp_config.name])
            self._temp_config_path = tmp_config.name  # Keep ref to delete later?

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create temp config: {e}")
            return

        # Switch to Console tab
        self.tabs.setCurrentIndex(1)

        # Start Process
        # We run 'prism' command. Assuming it's in path or we use current python.
        # Ideally use sys.executable + ["-m", "skyline_prism", "run", ...]
        cmd = sys.executable
        full_args = ["-m", "skyline_prism"] + args

        self.console_tab.start_process(cmd, full_args, output_dir)

    def _on_process_finished(self, success: bool, output_dir: str) -> None:
        """Handle completion of the analysis process."""
        if success:
            # Enable Results tab
            try:
                # Remove placeholder if present
                if isinstance(self.result_tab, QWidget) and not isinstance(
                    self.result_tab, PRISMResultWidget
                ):
                    idx = self.tabs.indexOf(self.result_tab)
                    if idx != -1:
                        self.tabs.removeTab(idx)
                        self.result_tab.deleteLater()

                # Create Result Widget
                self.result_tab = PRISMResultWidget(output_dir)
                self.tabs.addTab(self.result_tab, "Results")
                self.tabs.setTabEnabled(self.tabs.indexOf(self.result_tab), True)

                # Switch to it
                self.tabs.setCurrentWidget(self.result_tab)

            except Exception as e:
                self.console_tab.append_text(f"\nError initializing viewer: {e}\n", "red")
        else:
            # Stay on console tab
            pass


def main():
    app = QApplication(sys.argv)
    window = PRISMMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
