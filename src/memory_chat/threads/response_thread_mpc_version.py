from PySide6.QtCore import QThread, Signal
from amp_lib import OpenAIClient
from fastmcp_http.client import FastMCPHttpClient


class AIResponseThread(QThread):
    response_ready = Signal(str)

    def __init__(self, messages, model):
        super().__init__()
        # self.openai_client = OpenAIClient(base_url="http://127.0.0.1:17173", api_key="")
        self.fastmcp_client = FastMCPHttpClient()
        self.messages = messages
        self.model = model

    def run(self):
        print(self.messages)
        # response = self.openai_client.chat_completion(self.messages, self.model)
        # ai_message = response["choices"][0]["message"]["content"]
        # for each message remove timestamp
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in self.messages
        ]
        response = self.fastmcp_client.call_tool(
            "AMPServer.chat_completion",
            {"messages": messages, "model": self.model},
        )
        ai_message = response[0].text
        print("AI Response:", ai_message)
        self.response_ready.emit(ai_message)
