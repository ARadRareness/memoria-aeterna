import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging
from waitress import serve
import uuid
from datetime import datetime
import time
import yaml
import numpy as np

from src.memory_embeddings.stella_embeddings import StellaEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MemoryManager:
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

    def search(self, query):
        print("SEARCHING FOR", query)
        query_embedding = self.embedder.embed_query(query)
        similarities = self.embedder.similarity(query_embedding, self.embeddings, k=5)
        memories = [self.memories[i] for i in similarities]

        memory_strings = [
            f"ID: {memory['id']}\nTopic: {memory['topic']}\nContent: {memory['content']}"
            for memory in memories
        ]
        print("FOUND MEMORIES", memory_strings)
        return memories

    def filter_by_tags(self, tags):
        if not tags:
            return self.memories
        return [m for m in self.memories if any(tag in m["tags"] for tag in tags)]


app = Flask(__name__)

# Initialize the memory manager
memory_manager = MemoryManager()


@app.route("/add_memory", methods=["POST"])
def add_memory():
    start_time = time.time()
    try:
        data = request.json
        if not data or "topic" not in data:
            return jsonify({"error": "Missing 'topic' in request"}), 400

        memory_id = memory_manager.add(data)

        end_time = time.time()
        logger.info(f"add_memory operation took {end_time - start_time:.4f} seconds")
        return jsonify({"message": "Memory added successfully", "id": memory_id}), 201
    except Exception as e:
        logger.exception("Error in add_memory")
        return jsonify({"error": str(e)}), 500


@app.route("/retrieve_memories", methods=["GET"])
def retrieve_memories():
    try:
        tags = request.args.getlist("tag")
        filtered_memories = memory_manager.filter_by_tags(tags)
        return jsonify(filtered_memories), 200
    except Exception as e:
        logger.exception("Error in retrieve_memories")
        return jsonify({"error": str(e)}), 500


@app.route("/update_memory/<memory_id>", methods=["PUT"])
def update_memory(memory_id):
    try:
        data = request.json
        if memory_manager.update(memory_id, data):
            return jsonify({"message": "Memory updated successfully"}), 200
        return jsonify({"error": "Memory not found"}), 404
    except Exception as e:
        logger.exception("Error in update_memory")
        return jsonify({"error": str(e)}), 500


@app.route("/delete_memory/<memory_id>", methods=["DELETE"])
def delete_memory(memory_id):
    try:
        if memory_manager.delete(memory_id):
            return jsonify({"message": "Memory deleted successfully"}), 200
        return jsonify({"error": "Memory not found"}), 404
    except Exception as e:
        logger.exception("Error in delete_memory")
        return jsonify({"error": str(e)}), 500


@app.route("/search_memories", methods=["GET"])
def search_memories():
    try:
        query = request.args.get("q", "").lower()
        results = memory_manager.search(query)
        return jsonify(results), 200
    except Exception as e:
        logger.exception("Error in search_memories")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        port = int(os.getenv("SERVER_PORT", 17174))
        logger.info(f"Starting Memoria Aeterna on port {port}")
        serve(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {str(e)}")
