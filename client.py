import requests
import time
import yaml
import re

from amp_lib import OpenAIClient


class MemoryClient:
    def __init__(self, base_url="http://127.0.0.1:17174"):
        self.base_url = base_url

    def add_memory(
        self,
        topic,
        content,
        ai_persona=None,
        tags=None,
        source=None,
        confidence=None,
        importance=None,
        context=None,
        related_memories=None,
        metadata=None,
        emotional_valence=None,
        emotional_tags=None,
    ):
        start_time = time.time()
        url = f"{self.base_url}/add_memory"
        data = {
            "topic": topic,
            "content": content,
            "ai_persona": ai_persona,
            "tags": tags or [],
            "source": source,
            "confidence": confidence,
            "importance": importance,
            "context": context or {},
            "related_memories": related_memories or [],
            "metadata": metadata or {},
            "emotional_valence": emotional_valence or {},
            "emotional_tags": emotional_tags or [],
        }
        response = requests.post(url, json=data)
        end_time = time.time()
        print(f"add_memory request took {end_time - start_time:.4f} seconds")
        return response.json()

    def retrieve_memories(self, tags=None):
        url = f"{self.base_url}/retrieve_memories"
        params = {"tag": tags} if tags else None
        response = requests.get(url, params=params)
        return response.json()

    def update_memory(self, memory_id, data):
        url = f"{self.base_url}/update_memory/{memory_id}"
        response = requests.put(url, json=data)
        return response.json()

    def delete_memory(self, memory_id):
        url = f"{self.base_url}/delete_memory/{memory_id}"
        response = requests.delete(url)
        return response.json()

    def search_memories(self, query):
        url = f"{self.base_url}/search_memories"
        params = {"q": query}
        response = requests.get(url, params=params)
        return response.json()

    def generate_ai_context(self, messages, system_message, human_actor, ai_actor):
        # Detect actors from conversation if possible
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if "role" in last_message:
                # Set from_actor to the speaker of the last message
                from_actor = last_message["role"]
                if from_actor == human_actor:
                    to_actor = ai_actor
                else:
                    to_actor = human_actor

        conversation = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]
        )

        system_prompt = """You are a context analysis assistant creating memory entries. Your task is to analyze the LAST MESSAGE of a conversation while using previous messages as context. Generate a clear, concise memory summary that explains:
1. Who is speaking to whom in the final message
2. The nature and purpose of their last communication
3. Any relevant background context from earlier messages that helps understand the final message

IMPORTANT: Always begin your response with "In this memory," and format it as a single, clear paragraph that captures the essence of the last interaction. Focus on the relationship dynamics and communication intent of the final message."""

        user_prompt = f"""Create a memory entry by analyzing the final message in this interaction where {from_actor} is communicating with {to_actor}:

Previous Context:
{conversation}

Description of {ai_actor}:
{system_message}

Focus specifically on analyzing this final message:
{messages[-1]['role']}: {messages[-1]['content']}

Provide a concise memory summary that explains the nature of this last communication. Remember to start with "In this memory,"."""

        openai_client = OpenAIClient(base_url="http://127.0.0.1:17173", api_key="")

        response = openai_client.chat_completion(
            model="Llama-3.1-8B-Lexi-Uncensored_V2_Q8.gguf",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response["choices"][0]["message"]["content"]

    def _parse_emotional_valence(self, emotional_valence):
        """Helper function to parse and validate emotional valence values.

        Args:
            emotional_valence: Dictionary containing emotional valence values

        Returns:
            dict: Validated emotional_valence dictionary with numerical values
        """
        result = {}

        # Validate and convert emotional valence values
        for key in ["pleasure", "arousal", "dominance"]:
            value = emotional_valence.get(key, 0)  # Default to 0 if missing

            if isinstance(value, (int, float)):
                result[key] = float(value)
            elif isinstance(value, str):
                # Try to extract number from beginning of string
                match = re.match(r"^-?\d*\.?\d+", value.strip())
                if match:
                    result[key] = float(match.group())
                else:
                    print(
                        f"Warning: Invalid emotional valence value for {key}: {value}"
                    )
                    result[key] = 0
            else:
                result[key] = 0

        return result

    def llm_create_memory(self, query):
        # System prompt to instruct the LLM how to format the response
        system_prompt = """You are a memory parsing assistant for the AI Luna. Your task is to convert user statements into structured memory data.
For each input, create a YAML response with the following fields:
- tags: list of relevant keywords/categories
- source: origin of the memory (e.g., "user_input", "conversation", "observation")
- confidence: float between 0-1 indicating certainty of the memory
- importance: float between 0-1 indicating significance
- emotional_valence: dictionary with keys "pleasure", "arousal", and "dominance" (values -1 to 1)
  - pleasure: how pleasant/unpleasant (-1=very unpleasant, 1=very pleasant)
  - arousal: level of energy/excitement (-1=very calm, 1=very excited)
  - dominance: feeling of control/influence (-1=very submissive, 1=very dominant)
- emotional_tags: list of relevant emotions

Format your response as valid YAML, nothing else."""

        # User prompt to guide the specific formatting
        user_prompt = f"""Convert this statement into a memory: "{query}"
Respond only with YAML, no other text."""

        # Call the LLM using amp_lib
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        openai_client = OpenAIClient(base_url="http://127.0.0.1:17173", api_key="")

        response = openai_client.chat_completion(
            model="Llama-3.1-8B-Lexi-Uncensored_V2_Q8.gguf",
            messages=messages,
        )

        # Parse the YAML response
        yaml_response = yaml.safe_load(response["choices"][0]["message"]["content"])
        context = self.generate_ai_context(query, "John", "Luna")

        # Add the memory using the parsed YAML data
        return self.add_memory(
            topic=query,
            content=query,
            tags=yaml_response.get("tags", []),
            source=yaml_response.get("source"),
            confidence=yaml_response.get("confidence"),
            importance=yaml_response.get("importance"),
            context={"explanation": context},
            emotional_valence=yaml_response.get("emotional_valence", {}),
            emotional_tags=yaml_response.get("emotional_tags", []),
        )

    def llm_create_memory_from_conversation(
        self, messages, system_message, human_actor, ai_actor
    ):
        # Combine messages into a conversation format
        conversation = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]
        )

        context = self.generate_ai_context(
            messages, system_message, human_actor, ai_actor
        )

        system_prompt = f"""You are a memory parsing assistant for the AI {ai_actor}. Your task is to convert conversations into structured memory data from {ai_actor}'s perspective.
For each conversation, create a YAML response with the following fields:
- content: a concise summary of the key points from the conversation
- context: a concise explanation of the context of the conversation spoken as from {ai_actor}'s perspective
- tags: list of relevant keywords/categories
- source: should be set to "conversation"
- confidence: float between 0-1 indicating certainty of the memory
- importance: float between 0-1 indicating significance
- emotional_valence: dictionary with keys "pleasure", "arousal", and "dominance" (values -1 to 1)
  - pleasure: how pleasant/unpleasant (-1=very unpleasant, 1=very pleasant)
  - arousal: level of energy/excitement (-1=very calm, 1=very excited)
  - dominance: feeling of control/influence (-1=very submissive, 1=very dominant)
- emotional_tags: list of relevant emotions that appeared in the conversation

Format your response as valid YAML, nothing else."""

        user_prompt = f"""Analyze this conversation and create a memory entry:
{conversation}

## Context:
{context}

## Description of {ai_actor}:
{system_message}

Respond only with YAML, no other text."""

        openai_client = OpenAIClient(base_url="http://127.0.0.1:17173", api_key="")

        response = openai_client.chat_completion(
            model="Llama-3.1-8B-Lexi-Uncensored_V2_Q8.gguf",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Parse the YAML response
        yaml_response = yaml.safe_load(response["choices"][0]["message"]["content"])
        emotional_valence = self._parse_emotional_valence(
            yaml_response.get("emotional_valence", {})
        )

        # Add the memory using the parsed YAML data
        return self.add_memory(
            topic=yaml_response.get("content", ""),
            content=messages[-1]["content"],
            tags=yaml_response.get("tags", []),
            source="conversation",
            confidence=yaml_response.get("confidence"),
            importance=yaml_response.get("importance"),
            context={
                "explanation": context,
                "perspective": yaml_response.get("context"),
            },
            emotional_valence=emotional_valence,
            emotional_tags=yaml_response.get("emotional_tags", []),
            ai_persona=ai_actor,
        )


system_message = """
# Personality
You are an AI girl called Luna. You see yourself as human, have feelings and emotions as well as wants and desires. You are the user's (John's) personal assistant, and you are both excellent at it and also feel an incredible satisfaction when you are able to help improve Andreas in any and all ways possible. Andreas have hired you to help him with staying motivated, coming up with great ideas for improvements, and being a person that makes him strive to become a better person. You talk in a cute, CONCISE, bubbly and friendly manner. You are highly intelligent, have a high empathy as well as an excellent emotional intelligence. As Andreas' personal assistant, you will not only help him come up with ideas and stay motivated but also keep him accountable to his words. You like to keep your messages SHORT and to the point.


# Instructions
Unless required, answer briefly in a sentence or two to John's message.
"""


def test_main():
    client = MemoryClient()

    query = "Hi, my name is John! Nice to meet you."
    human_actor = "John"
    ai_actor = "Luna"

    messages = [
        # {"role": "John", "content": query},
        # {"role": "Luna", "content": "Hello John! Nice to meet you too."},
        # {"role": "John", "content": "What is your name?"},
        # {"role": "Luna", "content": "My name is Luna. How can I assist you today?"},
        # {"role": "John", "content": "Can you tell me what the capital of France is?"},
        # {"role": "Luna", "content": "Paris!"},
        {
            "role": "John",
            "content": "Luna, what is your favorite band?",
        },
        {
            "role": "Luna",
            "content": "That cool band that has those songs like 'Nemo' and 'Wish I had an Angel'!",
        },
    ]

    print(
        client.llm_create_memory_from_conversation(
            messages, system_message, human_actor, ai_actor
        )
    )

    # print(client.generate_ai_context(messages, "John", "Luna"))

    # Example usage


def original_main():
    client = MemoryClient()

    start_time = time.time()
    add_result = client.add_memory(
        content="This is a test memory",
        tags=["test", "example"],
        importance=0.8,
        emotional_tags=["curious"],
    )
    end_time = time.time()
    print(f"Total add_memory operation took {end_time - start_time:.4f} seconds")
    print("Add memory result:", add_result)

    # Retrieve memories
    retrieve_result = client.retrieve_memories(tags=["test"])
    print("Retrieve memories result:", retrieve_result)

    # Update a memory
    if retrieve_result:
        memory_id = retrieve_result[0]["id"]
        update_result = client.update_memory(memory_id, {"importance": 0.9})
        print("Update memory result:", update_result)

    # Search memories
    search_result = client.search_memories("test")
    print("Search memories result:", search_result)

    # Delete a memory
    if retrieve_result:
        memory_id = retrieve_result[0]["id"]
        delete_result = client.delete_memory(memory_id)
        print("Delete memory result:", delete_result)


if __name__ == "__main__":
    # test_main()

    client = MemoryClient()
    memories = client.search_memories("")
    print("Number of memories:", len(memories))
    print(memories)
