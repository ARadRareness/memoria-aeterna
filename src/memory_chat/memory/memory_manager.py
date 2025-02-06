from typing import List, Dict, Optional
from src.memory_chat.chat_utils.memory_client import MemoryClient
from src.memory_chat.memory.memory_relevance_analyzer import MemoryRelevanceAnalyzer
from src.memory_chat.memory.memory_finder import MemoryFinder
from src.memory_chat.threads.memory_thread import MemoryThread
from amp_lib import OpenAIClient


class MemoryManager:
    def __init__(
        self,
        openai_client: OpenAIClient,
        use_memory: bool = False,
        access_memories: bool = False,
    ):
        self.memory_client = MemoryClient()
        self.memory_finder = MemoryFinder(self.memory_client, openai_client)
        self.memory_threads = []
        self.memory_relevance_analyzer = MemoryRelevanceAnalyzer(openai_client)
        self.use_memory = use_memory
        self.access_memories = access_memories

    def process_conversation_memory(
        self,
        messages: List[Dict],
        system_message: str,
        human_actor: str,
        ai_actor: str,
        ai_persona: str,
        model_name: str,
        openai_client: OpenAIClient,
    ) -> None:
        # Clean up finished threads
        self.memory_threads = [t for t in self.memory_threads if t.isRunning()]

        # Create and start new memory thread
        memory_thread = MemoryThread(
            messages=messages.copy(),
            system_message=system_message,
            human_actor=human_actor,
            ai_actor=ai_actor,
            ai_persona=ai_persona,
            model_name=model_name,
            memory_client=self.memory_client,
            openai_client=openai_client,
        )
        self.memory_threads.append(memory_thread)
        memory_thread.start()

    def recall_memories(self, conversation_history: List[Dict]) -> Optional[str]:
        memories = self.memory_finder.recall_memories(conversation_history)

        if not memories:
            return None

        relevant_memories = self.memory_relevance_analyzer.analyze_relevance(
            conversation_history=conversation_history, memories=memories
        )

        if relevant_memories:
            memory_texts = []
            for memory in relevant_memories:
                explanation = memory["context"]["explanation"]
                content = memory["content"]
                memory_texts.append(
                    f"Memory Timestamp: {memory['timestamp']}\nContent: {content}\nContext: {explanation}\n"
                )
            return "\n\n".join(memory_texts)
        return None

    def cleanup(self):
        for thread in self.memory_threads:
            thread.wait()
        self.memory_threads.clear()
