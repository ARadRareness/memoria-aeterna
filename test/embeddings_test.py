import warnings
import logging
import numpy as np
import pytest

# Filter out specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("xformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from src.memory_embeddings.stella_embeddings import StellaEmbeddings


@pytest.fixture
def embedder():
    return StellaEmbeddings()


@pytest.fixture
def docs():
    return [
        "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
        "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
        "Stress is a common problem that can affect anyone. It can be caused by a variety of factors, including work, relationships, and personal issues. Stress can lead to physical and mental health problems, so it's important to find ways to manage it effectively.",
        "I like to go to the gym",
        "I like to go to the movies",
    ]


@pytest.fixture
def query():
    return "What are some ways to reduce stress?"


def test_embed_query(embedder, query):
    query_embedding = embedder.embed_query(query)
    assert query_embedding is not None
    assert isinstance(query_embedding, np.ndarray)


def test_embed_docs(embedder, docs):
    doc_embeddings = embedder.embed_docs(docs)
    assert doc_embeddings is not None
    assert isinstance(doc_embeddings, np.ndarray)
    assert len(doc_embeddings) == len(docs)


def test_similarity(embedder, query, docs):
    query_embedding = embedder.embed_query(query)
    doc_embeddings = embedder.embed_docs(docs)
    similarities = embedder.similarity(query_embedding, doc_embeddings)

    assert similarities is not None
    assert len(similarities) == len(docs)
    assert all(0 <= sim <= 1 for sim in similarities)

    # Check if stress-related documents have higher similarity
    stress_indices = [0, 2]  # indices of docs about stress
    other_indices = [1, 3, 4]  # indices of docs not about stress

    avg_stress_similarity = np.mean([similarities[i] for i in stress_indices])
    avg_other_similarity = np.mean([similarities[i] for i in other_indices])

    assert avg_stress_similarity > avg_other_similarity
