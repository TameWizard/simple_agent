from typing import Optional, List

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class ProductDescription(BaseModel):
    id: str
    text: str
    meta: Optional[dict] = None
    score: Optional[float] = None


corpus = [
        {"id": "1",
         "text": "Wireless over-ear headphones with 30-hour battery life, noise-cancelling and foldable design.",
         "meta": {"tag": "electronics"}},
        {"id": "2",
         "text": "Insulated stainless-steel water bottle, 750ml, keeps drinks cold for 24 hours and leakproof lid.",
         "meta": {"tag": "kitchen"}},
        {"id": "3",
         "text": "Smart bedside lamp with warm-to-cool white spectrum, app-controlled scenes and sunrise alarm.",
         "meta": {"tag": "home_electronics"}},
        {"id": "4", "text": "Eco-friendly bamboo cutting board set with non-slip feet and built-in juice groove.",
         "meta": {"tag": "kitchen"}},
        {"id": "5",
         "text": "Compact portable charger (20,000 mAh) with dual USB-C ports and fast-charge support for phones and tablets.",
         "meta": {"tag": "accessories"}},
        {"id": "6", "text": "Handmade vegan soy candle, autumn spice scent, poured in a reusable glass jar.",
         "meta": {"tag": "home"}},
        {"id": "7",
         "text": "Cozy plush toy fox for toddlers, machine-washable with embroidered eyes and soft organic cotton.",
         "meta": {"tag": "kids"}},
        {"id": "8",
         "text": "Deluxe watercolor travel set with 24 pigments, refillable water brush, and magnetic mixing tray.",
         "meta": {"tag": "art_supplies"}},
        {"id": "9",
         "text": "Ergonomic memory-foam seat cushion that relieves lower back pressure and fits most office chairs.",
         "meta": {"tag": "office"}},
        {"id": "10",
         "text": "Board game for 2–6 players, cooperative mystery-solving with modular board and replayable scenarios.",
         "meta": {"tag": "games"}},
        {"id": "11",
         "text": "Organic herbal tea sampler: four blends in compostable sachets, calming and caffeine-free.",
         "meta": {"tag": "grocery"}},
        {"id": "12",
         "text": "Minimalist RFID-blocking wallet in vegetable-tanned leather with six card slots and a slim profile.",
         "meta": {"tag": "accessories"}},
    ]


class VectorDatabase:
    def __init__(self, corpus):
        if not isinstance(corpus, (list, tuple)):
            raise ValueError("corpus must be a list of dicts")

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = self.model.get_sentence_embedding_dimension()

        self.ids = [d["id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]
        self.metas = [d.get("meta", {}) for d in corpus]

        if len(self.texts) == 0:
            # allow constructing but searches will return empty
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            return

        # Ensure we get a numpy array and then normalize explicitly
        vectors = np.asarray(
            self.model.encode(self.texts, convert_to_tensor=False),
            dtype=np.float32,
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        self.vectors = vectors

    def search(self, query: str, top_k: int = 5) -> List[ProductDescription]:
        if self.vectors.shape[0] == 0:
            return []

        q_vec = np.asarray(
            self.model.encode(query, convert_to_tensor=False),
            dtype=np.float32,
        ).ravel()

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0
        q_vec = q_vec / q_norm

        # dot product because vectors are normalized
        sims = np.dot(self.vectors, q_vec)  # shape (n,)

        # handle top_k > n
        n = sims.shape[0]
        k = min(top_k, n)

        if k == n:
            idx_sorted = np.argsort(-sims)
        else:
            idx_part = np.argpartition(-sims, k - 1)[:k]
            idx_sorted = idx_part[np.argsort(-sims[idx_part])]

        return [
            ProductDescription(
                id=self.ids[i],
                text=self.texts[i],
                meta=dict(self.metas[i]),
                score=float(sims[i]),
            )
            for i in idx_sorted
        ]


vector_searcher = VectorDatabase(corpus)
