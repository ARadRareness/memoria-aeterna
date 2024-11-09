from PySide6.QtCore import QThread
from typing import List, Dict


class MemoryThread(QThread):
    def __init__(
        self,
        messages: List[Dict],
        system_message: str,
        human_actor: str,
        ai_actor: str,
        ai_persona: str,
        model_name: str,
        memory_client,
        openai_client,
    ):
        super().__init__()
        self.messages = messages
        self.system_message = system_message
        self.human_actor = human_actor
        self.ai_actor = ai_actor
        self.ai_persona = ai_persona
        self.model_name = model_name
        self.memory_client = memory_client
        self.openai_client = openai_client

    def run(self):
        try:
            print("Creating memory from user message")
            # Create memory from user message
            self.memory_client.llm_create_memory_from_conversation(
                messages=self.messages[:-1],
                system_message=self.system_message,
                human_actor=self.human_actor,
                ai_actor=self.ai_actor,
                ai_persona=self.ai_persona,
                open_ai_client=self.openai_client,
                model_name=self.model_name,
            )
            print("Memory created from user message")
            # Create memory from AI response
            self.memory_client.llm_create_memory_from_conversation(
                messages=self.messages,
                system_message=self.system_message,
                human_actor=self.human_actor,
                ai_actor=self.ai_actor,
                ai_persona=self.ai_persona,
                open_ai_client=self.openai_client,
                model_name=self.model_name,
            )
            print("Memory created from AI response")
        except Exception as e:
            print(f"Error creating memory: {e}")
