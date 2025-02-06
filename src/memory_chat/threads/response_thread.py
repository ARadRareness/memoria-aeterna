from PySide6.QtCore import QThread, Signal
from amp_lib import OpenAIClient


class AIResponseThread(QThread):
    response_ready = Signal(str)

    def __init__(self, messages, model, max_tokens=512):
        super().__init__()
        self.openai_client = OpenAIClient(base_url="http://127.0.0.1:17173", api_key="")
        self.messages = messages
        self.model = model
        self.max_tokens = max_tokens

    def run(self):
        response = self.openai_client.chat_completion(
            self.messages, self.model, self.max_tokens
        )
        ai_message = response["choices"][0]["message"]["content"]
        self.response_ready.emit(ai_message)
