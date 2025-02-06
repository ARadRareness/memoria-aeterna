from typing import Optional, List, Dict


class MemoryFinder:
    def __init__(self, memory_client, openai_client):
        self.memory_client = memory_client
        self.openai_client = openai_client

    def find_memories(self, query: str) -> Optional[str]:
        # Get all potentially relevant memories
        memories = self.memory_client.search_memories(query, k=5)
        return memories if memories else None

    def recall_memories(self, conversation_history: List[Dict]) -> Optional[str]:
        system_prompt = """You are an expert at identifying key topics and concepts from conversations.
Generate 2-3 search queries based on the conversation history, focusing on different aspects of the user's recent messages.
Focus on important subjects, names, events, or themes that would benefit from additional context.

IMPORTANT: Output ONLY the search queries, one per line. Do not include any explanations or additional text.

Full conversation history:
{conversation_history}

User's most recent message:
{latest_message}"""

        # Format conversation history and get latest user message
        formatted_conversation = []
        latest_user_message = None
        for msg in conversation_history:
            if msg["role"] != "system":
                formatted_conversation.append(
                    f"{msg['role'].title()}: {msg['content']}"
                )
                if msg["role"] == "user":
                    latest_user_message = msg["content"]

        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    conversation_history="\n".join(formatted_conversation),
                    latest_message=latest_user_message or "No user message found",
                ),
            },
            {
                "role": "user",
                "content": "Generate search queries based on the conversation, focusing on different aspects of the latest user message.",
            },
        ]

        # Get search queries from LLM
        response = self.openai_client.chat_completion(
            model="Llama-3.1-8B-Lexi-Uncensored_V2_Q8.gguf",
            messages=messages,
        )

        try:
            # Split response into multiple search queries
            search_queries = (
                response["choices"][0]["message"]["content"].strip().split("\n")
            )

            # Get memories for each query and combine results
            all_memories = []
            for query in search_queries:
                print(f"Search query: {query}")
                memories = self.find_memories(query)
                if memories:
                    all_memories.extend(memories)

            # Return combined memories or None if empty
            return all_memories if all_memories else None
        except (KeyError, IndexError) as e:
            print(f"Error generating search queries: {e}")
            return None
