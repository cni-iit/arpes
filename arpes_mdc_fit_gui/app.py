from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSplitter, QSlider, QGroupBox
)
from PySide6.QtCore import Qt
import sys


class ArpesFitMainWindow(QMainWindow):
    """
    Main application window for ARPES MDC extraction and fitting.
    This class only defines the GUI structure, not the fitting logic.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARPES MDC Fit Tool")
        self.resize(1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout
        main_layout = QVBoxLayout(central_widget)

        # --- Top controls: File loading ---
        file_layout = QHBoxLayout()

        self.load_button = QPushButton("Load HDF5 file")
        self.load_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_button)

        self.file_label = QLabel("No file loaded")
        self.file_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        file_layout.addWidget(self.file_label)

        main_layout.addLayout(file_layout)

        # --- Central splitter: preview + controls ---
        central_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(central_splitter)

        # ================= LEFT SIDE =================
        # Data preview area (ARPES map)
        self.preview_group = QGroupBox("ARPES Data Preview")
        preview_layout = QVBoxLayout(self.preview_group)

        # Placeholder for matplotlib canvas
        self.preview_placeholder = QLabel("[ARPES preview canvas]")
        self.preview_placeholder.setAlignment(Qt.AlignCenter)
        self.preview_placeholder.setStyleSheet("background-color: #222; color: #aaa;")
        preview_layout.addWidget(self.preview_placeholder)

        # Slider for 3D datasets (layer selection)
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setEnabled(False)
        preview_layout.addWidget(self.layer_slider)

        central_splitter.addWidget(self.preview_group)

        # ================= RIGHT SIDE =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        central_splitter.addWidget(right_panel)

        # -------- MDC extraction panel --------
        mdc_group = QGroupBox("MDC Extraction")
        mdc_layout = QVBoxLayout(mdc_group)

        self.mdc_placeholder = QLabel("[MDC plot]")
        self.mdc_placeholder.setAlignment(Qt.AlignCenter)
        self.mdc_placeholder.setStyleSheet("background-color: #222; color: #aaa;")
        mdc_layout.addWidget(self.mdc_placeholder)

        self.extract_mdc_button = QPushButton("Extract MDC")
        mdc_layout.addWidget(self.extract_mdc_button)

        right_layout.addWidget(mdc_group)

        # -------- Pre-processing panel --------
        preprocess_group = QGroupBox("Pre-processing")
        preprocess_layout = QVBoxLayout(preprocess_group)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setToolTip("Signal threshold")
        preprocess_layout.addWidget(QLabel("Intensity threshold"))
        preprocess_layout.addWidget(self.threshold_slider)

        right_layout.addWidget(preprocess_group)

        # -------- Fit model panel --------
        fit_group = QGroupBox("Fit Model")
        fit_layout = QVBoxLayout(fit_group)

        self.add_peak_button = QPushButton("Add Voigt peak")
        fit_layout.addWidget(self.add_peak_button)

        self.fit_button = QPushButton("Run fit")
        fit_layout.addWidget(self.fit_button)

        right_layout.addWidget(fit_group)
        right_layout.addStretch()

        # Enable drag and drop
        self.setAcceptDrops(True)

    # ================= Drag & Drop =================
    def dragEnterEvent(self, event):
        """Accept drag events if they contain files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle file drop."""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.load_file(path)

    # ================= Slots =================
    def load_file(self, path=None):
        """Open file dialog or load file from provided path."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open HDF5 File", "", "HDF5 files (*.h5 *.hdf5)"
            )
        if path:
            self.file_label.setText(path)
            # Actual loading logic will be implemented in the backend
            print(f"Loading file: {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ArpesFitMainWindow()
    window.show()
    sys.exit(app.exec())