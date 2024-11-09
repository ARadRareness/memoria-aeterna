import warnings
import logging

# Filter out specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("xformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer


class StellaEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            revision="2aa5579fcae1c579de199a3866b6e514bbbf5d10",
            trust_remote_code=True,
        ).cuda()

    def embed_docs(self, docs: list[str]):
        return self.model.encode(docs)

    def embed_query(self, query: str):
        # s2p_query, sentence-to-passage, for queries against documents
        # s2s_query, sentence-to-sentence, for semantic textual similarity task
        return self.model.encode(query, prompt_name="s2p_query")

    def similarity(self, query, doc_embeddings, k=3):
        results = self.model.similarity(query, doc_embeddings)
        print("RESULTS", results)
        # Retrieve the position of the top k results
        top_k_indices = results.squeeze().argsort(descending=True)[:k].tolist()
        return top_k_indices
