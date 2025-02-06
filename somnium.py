import json
from typing import List, Dict, Optional
from src.memory_utils.server_memory_manager import ServerMemoryManager
from amp_lib import OpenAIClient
import yaml
from datetime import datetime
import os
from tqdm import tqdm


class Somnium:
    def __init__(
        self, memory_manager: ServerMemoryManager, openai_client: OpenAIClient
    ):
        self.memory_manager = memory_manager
        self.openai_client = openai_client

    def load_dream_data(self) -> Optional[datetime]:
        """Load the timestamp of the last dream from last_dream.yaml"""
        last_dream_file = "last_dream.yaml"

        if os.path.exists(last_dream_file):
            with open(last_dream_file, "r") as f:
                timestamp_data = yaml.safe_load(f)
                if timestamp_data and "last_dream_timestamp" in timestamp_data:
                    return datetime.fromisoformat(
                        timestamp_data["last_dream_timestamp"]
                    )
        return None

    def save_dream_data(self):
        """Save the current timestamp as the last dream timestamp."""
        current_timestamp = datetime.now().isoformat()
        with open("last_dream.yaml", "w") as f:
            yaml.dump({"last_dream_timestamp": current_timestamp}, f)

    def dream(self):
        """Process memories by extracting tags and managing backups."""
        # Load last dream timestamp
        from_timestamp = self.load_dream_data()

        # Backup original memories before processing
        original_memories = self.memory_manager.memories.copy()

        # Extract tags from memories
        results = self.extract_tags(from_timestamp=from_timestamp)

        # If any memories were processed and tagged
        if results["processed_memories"] > 0:
            # Backup original memories
            self.backup_archived_memories(original_memories)

            # Save updated memories with new tags
            self.save_active_memories(self.memory_manager.memories)

        # Save current timestamp
        self.save_dream_data()

        return results

    def backup_archived_memories(self, archived_memories: List[Dict]):
        """Save archived memories to a backup file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = "memory_backups"
        os.makedirs(backup_dir, exist_ok=True)

        backup_path = os.path.join(backup_dir, f"archived_memories_{timestamp}.yaml")
        with open(backup_path, "w") as f:
            yaml.dump(archived_memories, f)

    def save_active_memories(self, active_memories: List[Dict]):
        """Save active memories to memories.yaml."""
        with open("memories.yaml", "w") as f:
            yaml.dump(active_memories, f)
        pass  # TODO: Implement this

    def extract_tags(self, from_timestamp: Optional[datetime] = None):
        """Main method to analyze memories and extract relevant tags using LLM."""
        # Load all memories
        all_memories = self.memory_manager.memories

        # Filter memories after from_timestamp
        if from_timestamp:
            all_memories = [
                memory
                for memory in all_memories
                if datetime.fromisoformat(memory["timestamp"]) > from_timestamp
            ]
            if not all_memories:
                print("No memories to process")
                return {
                    "total_memories": len(all_memories),
                    "processed_memories": 0,
                    "tagged_memories": [],
                }

        system_prompt = {
            "role": "system",
            "content": """You are a memory analysis assistant. For each memory, extract and expand relevant tags that categorize and describe the key elements of the memory using both existing tags and the memory content, context, and explanation.
            The information will be used to cluster similar memories, so be thorough and include all relevant tags that are not already present.
            Consider:
            - Main topics and subjects
            - Key actions or events
            
            Format your response as a JSON array of strings, containing only the most relevant tags. Only write out the json array, nothing else.
            Example: ["programming", "programming language","python", "debugging", "learning"]""",
        }

        tagged_memories = []
        for memory in tqdm(all_memories, desc="Extracting tags"):
            messages = [
                system_prompt,
                {
                    "role": "user",
                    "content": f"Please analyze this memory and provide relevant tags:\n{self.memory_manager.get_memory_full_text(memory)}",
                },
            ]

            response = self.openai_client.chat_completion(
                messages=messages, model="Llama-3.1-8B-Lexi-Uncensored_V2_Q8.gguf"
            )

            # Extract tags from response
            try:
                tags = json.loads(response["choices"][0]["message"]["content"])
                tags = [tag.replace("_", " ").lower().strip() for tag in tags]
                memory["tags"] = tags
                tagged_memories.append(memory)
            except (KeyError, IndexError) as e:
                print(f"Error processing memory {memory.get('id', 'unknown')}: {e}")
                continue

        self.save_active_memories(self.memory_manager.memories)

        return {
            "total_memories": len(all_memories),
            "processed_memories": len(tagged_memories),
            "tagged_memories": tagged_memories,
        }


def main():
    memory_manager = ServerMemoryManager()
    openai_client = OpenAIClient(api_key="", base_url="http://127.0.0.1:17173")

    dream_manager = Somnium(memory_manager, openai_client)
    results = dream_manager.dream()

    print(f"Total memories processed: {results['total_memories']}")


if __name__ == "__main__":
    main()
