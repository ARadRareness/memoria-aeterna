from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QFileDialog,
)


class SystemMessageDialog(QDialog):
    def __init__(self, current_message: str, current_persona: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit System Message")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # AI Persona input
        self.persona_edit = QTextEdit()
        self.persona_edit.setPlaceholderText("Enter AI Persona name...")
        self.persona_edit.setMaximumHeight(50)  # Limit height for persona input
        self.persona_edit.setPlainText(current_persona)  # Set default value
        layout.addWidget(self.persona_edit)

        # System message editor
        self.message_edit = QTextEdit()
        self.message_edit.setPlainText(current_message)
        layout.addWidget(self.message_edit)

        # Buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load from File")
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")

        # Set minimum width for load button
        load_button.setMinimumWidth(100)  # Adjust this value as needed

        load_button.clicked.connect(self.load_from_file)
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(load_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_message(self) -> str:
        return self.message_edit.toPlainText()

    def get_persona(self) -> str:
        return self.persona_edit.toPlainText()

    def load_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load System Message",
            "",
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*.*)",
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    self.message_edit.setPlainText(content)
            except Exception as e:
                print(f"Error loading file: {e}")
