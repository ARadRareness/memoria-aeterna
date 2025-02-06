from datetime import datetime
import logging
import yaml
import os
import uuid
import numpy as np
from src.memory_embeddings.stella_embeddings import StellaEmbeddings

logger = logging.getLogger(__name__)


class ServerMemoryManager:
    def __init__(self, file_path="memories.yaml"):
        self.file_path = file_path
        self.memories = self.load()
        self.embedder = StellaEmbeddings()
        self.embeddings = self.embed_all_documents()
        logger.info(f"Loaded {len(self.memories)} memories from {self.file_path}")

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                return yaml.safe_load(f) or []
        return []

    def save(self):
        with open(self.file_path, "w") as f:
            yaml.dump(
                self.memories,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

    def add(self, data):
        memory = {
            "id": str(uuid.uuid4()),
            "ai_persona": data.get("ai_persona", ""),
            "topic": data.get("topic", ""),
            "content": data.get("content", ""),
            "timestamp": datetime.now().isoformat(),
            "tags": data.get("tags", []),
            "source": data.get("source", "user"),
            "confidence": data.get("confidence", 1.0),
            "importance": data.get("importance", 0.5),
            "context": data.get("context", {}),
            "related_memories": data.get("related_memories", []),
            "last_accessed": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "access_count": 0,
            "modified_count": 0,
            "version": 1,
            "embedding": None,
            "metadata": data.get("metadata", {}),
            "emotional_valence": {
                "pleasure": data.get("emotional_valence", {}).get("pleasure", 0.0),
                "arousal": data.get("emotional_valence", {}).get("arousal", 0.0),
                "dominance": data.get("emotional_valence", {}).get("dominance", 0.0),
            },
            "emotional_tags": data.get("emotional_tags", []),
        }
        self.memories.append(memory)
        self.add_embedding(memory)

        self.save()
        return memory["id"]

    def unwrap_list(self, list_to_unwrap):
        elements = []
        for element in list_to_unwrap:
            if isinstance(element, dict):
                for key, value in element.items():
                    elements.append(f"{key}: {value}")
            else:
                elements.append(element)
        return elements

    def get_memory_full_text(self, memory):
        """Helper method to combine all relevant memory information into a single text."""
        components = [
            "Content: " + memory["content"],
            "Context: " + memory["context"].get("explanation", ""),
            "Topic: " + memory["topic"] if memory["topic"] else "",
            (
                "Tags: " + ", ".join(self.unwrap_list(memory["tags"]))
                if memory["tags"]
                else ""
            ),
            "AI Persona: " + memory["ai_persona"] if memory["ai_persona"] else "",
            "Timestamp: " + memory["timestamp"] if memory["timestamp"] else "",
        ]
        return "\n".join(filter(None, components))

    def embed_all_documents(self):
        return self.embedder.embed_docs(
            [
                memory["content"] + "\n" + memory["context"]["explanation"]
                for memory in self.memories
            ]
        )

    def add_embedding(self, memory):
        new_embedding = self.embedder.embed_docs(
            [memory["content"] + "\n" + memory["context"]["explanation"]]
        )
        self.embeddings = (
            np.vstack([self.embeddings, new_embedding])
            if len(self.embeddings) > 0
            else new_embedding
        )

    def update(self, memory_id, data):
        for i, memory in enumerate(self.memories):
            if memory["id"] == memory_id:
                memory.update(data)
                self.update_embedding(memory, i)
                self.save()
                return True
        return False

    def update_embedding(self, memory, i):
        new_embedding = self.embedder.embed_docs(
            [memory["content"] + "\n" + memory["context"]["explanation"]]
        )
        self.embeddings[i] = new_embedding

    def delete(self, memory_id):
        delete_index = next(
            (i for i, m in enumerate(self.memories) if m["id"] == memory_id), -1
        )
        if delete_index != -1:
            self.memories.pop(delete_index)
            self.delete_embedding(delete_index)
            self.save()
            return True
        return False

    def delete_embedding(self, delete_index):
        self.embeddings = np.delete(self.embeddings, delete_index, axis=0)

    def search(self, query, k=10):
        print("SEARCHING FOR", query)
        query_embedding = self.embedder.embed_query(query)
        similarities = self.embedder.similarity(query_embedding, self.embeddings, k=k)
        memories = [self.memories[i] for i in similarities]

        memory_strings = [
            f"ID: {memory['id']}\nTopic: {memory['topic']}\nContent: {memory['content']}"
            for memory in memories
        ]
        # print("FOUND MEMORIES", memory_strings)
        return memories

    def filter_by_tags(self, tags):
        if not tags:
            return self.memories
        return [m for m in self.memories if any(tag in m["tags"] for tag in tags)]
