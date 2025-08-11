from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = ["Doc text 1...", "Doc text 2..."]
embs = model.encode(docs, convert_to_numpy=True)
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)
faiss.write_index(index, "docs.index")

with open("meta.pkl", "wb") as f:
	pickle.dump(docs, f)
