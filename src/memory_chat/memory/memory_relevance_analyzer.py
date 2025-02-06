from typing import List, Dict
import json


class MemoryRelevanceAnalyzer:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.system_prompt = """You are an expert at determining if memories are relevant to mention to the AI assistant based on the current conversation.
Your task is to analyze the conversation history and the provided memories, then determine which memories are truly relevant and would add value to the conversation.
Only select memories that are directly related to the current topic or would provide meaningful context to the AI's response.

Conversation history:
{conversation_history}

Available memories:
{memories}

Respond with a JSON array containing only the indices of relevant memories (0-based). For example: [0, 2] means only the first and third memories are relevant.
If no memories are relevant, respond with an empty array []."""

    def analyze_relevance(
        self,
        conversation_history: List[Dict],
        memories: List[Dict],
    ) -> List[Dict]:
        if not memories:
            return []

        # Format memories with indices
        formatted_memories = []
        for i, memory in enumerate(memories):
            formatted_memories.append(
                f"[{i}] Content: {memory['content']}\nContext: {memory['context']['explanation']}"
            )

        # Format conversation history
        formatted_conversation = []
        for msg in conversation_history:
            if msg["role"] != "system":
                formatted_conversation.append(
                    f"{msg['role'].title()}: {msg['content']}"
                )

        # print("Formatted memories:", formatted_memories)

        # Prepare the messages for the LLM
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(
                    conversation_history="\n".join(formatted_conversation),
                    memories="\n\n".join(formatted_memories),
                ),
            },
            {
                "role": "user",
                "content": "Output only the array of relevant memory indices.",
            },
        ]

        # Get response from LLM
        response = self.openai_client.chat_completion(
            model="Llama-3.1-8B-Lexi-Uncensored_V2_Q8.gguf",  # or whatever model you're using
            messages=messages,
        )

        try:
            # Parse the response to get relevant indices
            relevant_indices = json.loads(response["choices"][0]["message"]["content"])
            # Return only the relevant memories
            return [memories[i] for i in relevant_indices]
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing relevance response: {e}")
            return []
