import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging
from waitress import serve

import time

from src.memory_utils.server_memory_manager import ServerMemoryManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


app = Flask(__name__)

# Initialize the memory manager
memory_manager = ServerMemoryManager()


@app.route("/add_memory", methods=["POST"])
def add_memory():
    print("STARTING ADD MEMORY")
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
        k = int(request.args.get("k", 10))  # Default to 10 if not specified
        results = memory_manager.search(query, k=k)
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
