import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import threading
import time
import queue

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QComboBox,
    QPushButton,
    QLabel,
    QFileDialog,
)
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat, QAction
from PySide6.QtCore import Qt, QThread, Signal

from amp_lib import AmpClient
from amp_lib import OpenAIClient
from src.memory_chat.chat_utils.voice_input import VoiceInput
from src.memory_chat.threads.tts_thread import TTSThread
from src.memory_chat.gui.system_message_dialog import SystemMessageDialog
from src.memory_chat.threads.response_thread import AIResponseThread
from src.memory_chat.chat_utils.memory_client import MemoryClient
from src.memory_chat.threads.memory_thread import MemoryThread


class VoiceInputThread(QThread):
    message_ready = Signal(str)

    def __init__(self, voice_input, amp_client, use_local_whisper):
        super().__init__()
        self.voice_input = voice_input
        self.amp_client = amp_client
        self.use_local_whisper = use_local_whisper
        self.is_running = True
        self.single_record = False

    def run(self):
        while self.is_running:
            try:
                time.sleep(0.1)
                message = None
                with threading.Lock():
                    if self.voice_input:
                        message = self.voice_input.get_input()

                if message:
                    if not self.use_local_whisper:
                        message = self.amp_client.speech_to_text(
                            audio_file_path=message
                        )
                        print("Speech to text:", message)

                    if message:
                        self.message_ready.emit(message)

                        if self.single_record:
                            self.is_running = False

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing voice input: {e}")
                break

    def stop(self):
        self.is_running = False


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.openai_client = OpenAIClient(base_url="http://127.0.0.1:17173", api_key="")
        self.amp_client = AmpClient(base_url="http://127.0.0.1:17173")
        self.current_conversation_id = None
        self.ai_persona = "assistant"
        self.system_message = "You are a helpful AI assistant."
        self.messages = []
        self.use_tts = False
        self.voice_input = None
        self.voice_input_thread = None
        self.allow_voice_interrupt = False
        self.is_closing = False
        self.use_local_whisper = False
        self.use_memory = True
        self.access_memories = False
        self.memory_client = MemoryClient()
        self.memory_threads = []

        self.init_ui()
        self.load_or_create_conversation()

    def init_ui(self):
        self.setWindowTitle("Memory Chat")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Top controls
        top_controls = QHBoxLayout()

        # Model selector
        self.model_selector = QComboBox()
        models = self.get_available_models()
        self.model_selector.addItems(models)

        top_controls.addWidget(QLabel("Model:"))
        top_controls.addWidget(self.model_selector)

        # System message button
        system_msg_btn = QPushButton("Edit System Message")
        system_msg_btn.clicked.connect(self.edit_system_message)
        top_controls.addWidget(system_msg_btn)

        # Load conversation button
        load_btn = QPushButton("Load Conversation")
        load_btn.clicked.connect(self.load_conversation)
        top_controls.addWidget(load_btn)

        layout.addLayout(top_controls)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setStyleSheet(
            "background-color: #f7e9ef;"
        )  # Light pink background
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # Input area
        self.input_box = QTextEdit()
        self.input_box.setMaximumHeight(100)
        self.input_box.keyPressEvent = self.handle_input_keys
        layout.addWidget(self.input_box)

        # Add Options menu
        menu_bar = self.menuBar()
        options_menu = menu_bar.addMenu("Options")

        # Add New Chat action
        new_chat_action = QAction("New Chat", self)
        new_chat_action.triggered.connect(self.start_new_chat)
        options_menu.addAction(new_chat_action)

        # Add TTS toggle action
        self.tts_action = QAction("Use Text to Speech", self)
        self.tts_action.setCheckable(True)
        self.tts_action.triggered.connect(self.toggle_tts)
        options_menu.addAction(self.tts_action)

        # Add voice input toggle action
        self.use_voice_input_action = QAction("Use Voice Input", self)
        self.use_voice_input_action.setCheckable(True)
        self.use_voice_input_action.triggered.connect(self.toggle_voice_input)
        options_menu.addAction(self.use_voice_input_action)

        # Add record audio button
        self.record_audio_button = QPushButton("Record Audio")
        self.record_audio_button.clicked.connect(self.record_audio)
        layout.addWidget(self.record_audio_button)

        # Add memory toggle action to Options menu
        self.use_memory_action = QAction("Create Memories", self)
        self.use_memory_action.setCheckable(True)
        self.use_memory_action.setChecked(True)
        self.use_memory_action.triggered.connect(self.toggle_memory)
        options_menu.addAction(self.use_memory_action)

        # Add access memories toggle action
        self.access_memories_action = QAction("Access Memories", self)
        self.access_memories_action.setCheckable(True)
        self.access_memories_action.setChecked(False)
        self.access_memories_action.triggered.connect(self.toggle_access_memories)
        options_menu.addAction(self.access_memories_action)

    def get_available_models(self) -> List[str]:
        try:
            response = self.openai_client.models.list()
            return [model.id for model in response.data]
        except:
            return ["default-model"]  # Fallback if server unavailable

    def handle_input_keys(self, event):
        if event.key() == Qt.Key_Return and event.modifiers() != Qt.ShiftModifier:
            self.send_message()
        else:
            QTextEdit.keyPressEvent(self.input_box, event)

    def send_message(self):
        message = self.input_box.toPlainText().strip()
        if not message:
            return

        # Display user message
        self.display_message(message, is_user=True)
        self.input_box.clear()

        memory_information = None
        if self.access_memories:
            memory_information = self.recall_memories(message)

        messages = self.messages.copy()

        if memory_information:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_message
                    + f"\n\nFollowing are some of your memories, Only mention them if they are directly relevant to the current topic. If the memories are not specifically related to the current conversation, simply ignore them and respond to the user's message normally.[AI MEMORY RECALL]\n{memory_information}\n[END MEMORY RECALL]",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.messages.append(
            {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Create and start response thread
        self.response_thread = AIResponseThread(
            messages, self.model_selector.currentText()
        )
        self.response_thread.response_ready.connect(self.handle_ai_response)
        self.response_thread.start()

    def handle_ai_response(self, ai_message):
        self.display_message(ai_message, is_user=False)
        self.messages.append(
            {
                "role": "assistant",
                "content": ai_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Process memory if enabled
        if self.use_memory:
            # Clean up finished threads
            self.memory_threads = [t for t in self.memory_threads if t.isRunning()]

            # Create and start new memory thread
            memory_thread = MemoryThread(
                messages=self.messages.copy(),
                system_message=self.system_message,
                human_actor="user",
                ai_actor="assistant",
                ai_persona=self.ai_persona,
                model_name=self.model_selector.currentText(),
                memory_client=self.memory_client,
                openai_client=self.openai_client,
            )
            memory_thread.finished.connect(
                lambda: self.cleanup_memory_thread(memory_thread)
            )
            self.memory_threads.append(memory_thread)
            memory_thread.start()

        # Add TTS playback if enabled
        if self.use_tts:
            self.play_tts(ai_message)

        # After AI response, if voice input is enabled, return to listening state
        if (
            hasattr(self, "use_voice_input_action")
            and self.use_voice_input_action.isChecked()
        ):
            self.update_record_button_state_listening()
        else:
            self.update_record_button_state_record_audio()

        self.save_conversation()

    def display_message(self, message: str, is_user: bool):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Create format with color
        format = QTextCharFormat()
        format.setForeground(QColor("darkblue" if is_user else "darkred"))

        # Apply the format and insert text
        cursor.setCharFormat(format)
        prefix = "User: " if is_user else "AI: "
        cursor.insertText(f"{prefix}{message}\n\n")

        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def edit_system_message(self):
        dialog = SystemMessageDialog(self.system_message, self.ai_persona, self)
        if dialog.exec():
            self.system_message = dialog.get_message()
            self.ai_persona = dialog.get_persona()
            # Append the new system message with timestamp
            self.messages.append(
                {
                    "role": "system",
                    "content": self.system_message,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.save_conversation()

    def get_conversation_path(self, conversation_id: Optional[str] = None) -> Path:
        base_path = Path("conversations")
        base_path.mkdir(exist_ok=True)

        if conversation_id:
            return base_path / f"{conversation_id}.json"
        return base_path

    def save_conversation(self):
        if not self.current_conversation_id:
            return

        conversation_data = {
            "id": self.current_conversation_id,
            "system_message": self.system_message,
            "ai_persona": self.ai_persona,
            "model": self.model_selector.currentText(),
            "messages": self.messages,  # Store the actual messages list instead of display text
            "timestamp": datetime.now().isoformat(),
        }

        file_path = self.get_conversation_path(self.current_conversation_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, indent=2)

    def load_conversation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Conversation",
            str(self.get_conversation_path()),
            "JSON files (*.json)",
        )

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract AI persona from filename
            self.current_conversation_id = data["id"]
            self.system_message = data["system_message"]
            self.ai_persona = data.get("ai_persona", "assistant")
            self.model_selector.setCurrentText(data["model"])

            # Clear chat display
            self.chat_display.clear()

            # Load messages directly and display them
            self.messages = data["messages"]

            # Display all messages except system messages
            for message in self.messages:
                if message["role"] == "user":
                    self.display_message(message["content"], is_user=True)
                elif message["role"] == "assistant":
                    self.display_message(message["content"], is_user=False)

    def load_or_create_conversation(self):
        self.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages.append(
            {
                "role": "system",
                "content": self.system_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def toggle_tts(self, checked: bool):
        self.use_tts = checked

    def play_tts(self, text: str):
        self.tts_thread = TTSThread(text, self.amp_client)
        self.tts_thread.start()

    def toggle_voice_input(self, checked: bool):
        self.clear_voice_input()

        if checked:
            self.voice_input = VoiceInput(use_local_whisper=False)
            self.voice_input_thread = VoiceInputThread(
                self.voice_input, self.amp_client, self.use_local_whisper
            )
            self.voice_input_thread.message_ready.connect(self.handle_voice_message)
            self.voice_input_thread.start()
            self.update_record_button_state_listening()
        else:
            self.update_record_button_state_record_audio()

    def handle_voice_message(self, message):
        self.input_box.setPlainText(message)
        self.send_message()

        # Instead, if voice input is still enabled, return to listening state
        if self.use_voice_input_action.isChecked():
            self.update_record_button_state_listening()
        else:
            self.update_record_button_state_record_audio()
            self.clear_voice_input()

    def clear_voice_input(self):
        if hasattr(self, "voice_input_thread") and self.voice_input_thread:
            self.voice_input_thread.stop()
            self.voice_input_thread.wait()
            self.voice_input_thread = None

        if hasattr(self, "voice_input"):
            del self.voice_input

    def update_record_button_state_waiting(self):
        self.record_audio_button.setText("Waiting...")
        if hasattr(self, "voice_input"):
            self.voice_input.set_ignore_audio(True)
        self.record_audio_button.setEnabled(False)

    def update_record_button_state_listening(self):
        self.record_audio_button.setText("Listening...")
        if hasattr(self, "voice_input"):
            self.voice_input.set_ignore_audio(False)
        self.record_audio_button.setEnabled(False)

    def update_record_button_state_record_audio(self):
        self.record_audio_button.setText("Record Audio")
        self.record_audio_button.setEnabled(True)

    def record_audio(self):
        self.clear_voice_input()

        self.voice_input = VoiceInput(use_local_whisper=False)
        self.voice_input_thread = VoiceInputThread(
            self.voice_input, self.amp_client, self.use_local_whisper
        )
        self.voice_input_thread.single_record = True
        self.voice_input_thread.message_ready.connect(self.handle_voice_message)
        self.voice_input_thread.start()
        self.update_record_button_state_listening()

    def cleanup_memory_thread(self, thread):
        if thread in self.memory_threads:
            self.memory_threads.remove(thread)

    def cleanup_before_exit(self):
        self.clear_voice_input()
        for thread in self.memory_threads:
            thread.wait()
        self.memory_threads.clear()

    def closeEvent(self, event):
        """Handle window close event"""
        self.is_closing = True
        self.cleanup_before_exit()
        super().closeEvent(event)

    def start_new_chat(self):
        # Clear the chat display and messages
        self.chat_display.clear()
        self.messages = []

        # Create a new conversation ID and add system message
        self.load_or_create_conversation()

    def toggle_memory(self, checked: bool):
        self.use_memory = checked

    def toggle_access_memories(self, checked: bool):
        self.access_memories = checked

    def recall_memories(self, user_message: str):
        # Placeholder for memory recall functionality
        memories = self.memory_client.search_memories(user_message)

        if memories:
            memory_texts = []
            for memory in memories:
                explanation = memory["context"]["explanation"]
                content = memory["content"]
                memory_texts.append(f"Content: {content}\nContext: {explanation}")
            return "\n\n".join(memory_texts)

        return None
