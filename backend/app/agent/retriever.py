import faiss
import pickle
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os

class Retriever:
	def __init__(self, index_path: str, meta_path: str, model_name: str = "all-MiniLM-L6-v2"):
		if not os.path.exists(index_path):
			raise FileNotFoundError(f"FAISS index file not found at {index_path}")
		if not os.path.exists(meta_path):
			raise FileNotFoundError(f"Metadata file not found at {meta_path}")

		self.logger = logging.getLogger(__name__)
		self.model = SentenceTransformer(model_name)
		self.index = faiss.read_index(index_path)

		if not isinstance(self.index, faiss.IndexIDMap):
			self.index = faiss.IndexIDMap(self.index)

		with open(meta_path, "rb") as f:
			self.docs: List[str] = pickle.load(f)

		if self.index.ntotal != len(self.docs):
			raise ValueError("Mismatch between number of documents and FAISS index entries")

	def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.7) -> List[Tuple[str, float]]:
		""""
		Retrieve top_k documents similar to the query with similarity above threshold.

		Args:
            query (str): user query string.
            top_k (int): number of results to retrieve.
            threshold (float): similarity threshold cutoff (lower means more similar for L2).


		Returns a list of tuples (document, similarity_score).
		"""
		
		q_emd = self.model.encode([query], convert_to_numpy = True)
		q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)

		# Search FAISS index
		distances, indices = self.index.search(q_emb, top_k)

		results = []
		for dist, idx in zip(distances[0], indices[0]):
			cos_sim = 1 - dist / 2  # Convert L2 distance to cosine similarity
			if cos_sim >= threshold and 0 <= idx < len(self.docs):
				results.append((self.docs[idx], cos_sim))
		return results

if __name__ == "__main__":
	import sys
	logging.basicConfig(level=logging.DEBUG)
	retriever = Retriever("docs.index", "meta.pkl")
	query = "What is the process for onboarding new employees?" if len(sys.argv) < 2 else sys.argv[1]
	results = retriever.retrieve(query, top_k=5, threshold=0.7)
	for doc, score in results:
		print(f"Document: {doc[:100]}... | Similarity: {score:.4f}")