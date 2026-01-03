"""PRISM Console Widget for the Unified GUI.

Provides a widget to display running process output and log messages.
"""

from __future__ import annotations

from PyQt6.QtCore import QProcess, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QProgressBar, QVBoxLayout, QWidget


class PRISMConsoleWidget(QWidget):
    """Widget for displaying process output and status."""

    # Signal emitted when process finishes successfully
    process_finished = pyqtSignal(bool, str)  # success, output_dir

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._output_dir: str = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.status_label)

        # Progress Bar (indeterminate during run)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Console Text Area
        self.output_area = QPlainTextEdit()
        self.output_area.setReadOnly(True)
        # Use a monospace font
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.output_area.setFont(font)
        # Black background, green text for "hacker" feel or just clean console look?
        # Let's stick to standard system theme for now, maybe slight dark grey if requested later.
        layout.addWidget(self.output_area)

    def start_process(self, command: str, args: list[str], output_dir: str) -> None:
        """Start a subprocess and capture its output."""
        self._output_dir = output_dir
        self.output_area.clear()
        self.append_text(f"Starting command: {command} {' '.join(args)}\n", color="blue")

        self.status_label.setText("Running PRISM Analysis...")
        self.progress_bar.show()

        self._process = QProcess(self)
        self._process.setProgram(command)
        self._process.setArguments(args)

        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.finished.connect(self._handle_finished)
        self._process.errorOccurred.connect(self._handle_error)

        self._process.start()

    def stop_process(self) -> None:
        """Terminate the running process."""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            self.append_text("\nProcess killed by user.\n", color="red")

    def append_text(self, text: str, color: str | None = None) -> None:
        """Append text to the console output."""
        # Check current scroll position
        scrollbar = self.output_area.verticalScrollBar()
        was_at_bottom = scrollbar.value() == scrollbar.maximum()

        # Insert text
        if color:
            # HTML format for color
            # Replace newlines with <br> for HTML if strictly needed,
            # but appendHtml handles blocks. appendPlainText handles raw.
            # Using insertHtml at end is safer for specific coloring.
            cursor = self.output_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            format_html = f'<span style="color:{color}">{text.replace(chr(10), "<br>")}</span>'
            self.output_area.appendHtml(format_html)
        else:
            self.output_area.insertPlainText(text)

        # Auto-scroll if we were at bottom
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _handle_stdout(self) -> None:
        if self._process:
            data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
            self.append_text(data)

    def _handle_stderr(self) -> None:
        if self._process:
            data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
            self.append_text(data, color="red")

    def _handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.progress_bar.hide()
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self.status_label.setText("Analysis Completed Successfully")
            self.append_text("\nProcess finished successfully.\n", color="green")
            self.process_finished.emit(True, self._output_dir)
        else:
            self.status_label.setText("Analysis Failed")
            self.append_text(f"\nProcess failed with exit code {exit_code}.\n", color="red")
            self.process_finished.emit(False, self._output_dir)

        self._process = None

    def _handle_error(self, error: QProcess.ProcessError) -> None:
        self.progress_bar.hide()
        self.status_label.setText(f"Process Error: {error}")
        self.append_text(f"\nProcess failed to start or crashed: {error}\n", color="red")
        self.process_finished.emit(False, self._output_dir)
        self._process = None
