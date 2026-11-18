import glob
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List

import numpy as np

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "patches",
)

SEMANTIC_SYNONYMS = {
    "nerf": ["reduced", "increased", "delay", "windup", "cost"],
    "buff": ["increased", "free", "recharge", "through", "walls"],
    "ult": ["ultimate", "points", "neural", "theft", "thrash", "fury", "blade", "storm", "shadows"],
    "wire": ["trapwire", "tether", "concusses", "cypher"],
    "tripwire": ["trapwire", "tether", "cypher", "windup"],
    "flash": ["curveball", "blind", "paranoia", "nearsights", "guiding", "dizzy"],
    "molly": ["hot", "hands", "damage", "delay", "snake", "bite"],
    "smoke": ["cloudburst", "dark", "cover", "cyber", "cage", "vision"],
    "dash": ["tailwind", "entry", "jett"],
    "drone": ["owl", "recon", "tag", "ping", "sova"],
    "dart": ["recon", "shock", "bolt", "spycam", "sova"],
    "pillar": ["crate", "box", "sunset", "spammable", "central"],
    "pool": ["rotation", "competitive", "pearl", "split", "icebox", "lotus"],
    "eco": ["economy", "credits", "save", "buy", "shields", "pistol"],
    "defuse": ["half", "spike", "post", "plant", "staggered", "delay"],
    "wall": ["blaze", "terrain", "paranoia", "penetration"],
    "clone": ["fakeout", "decoy", "yoru"],
    "orb": ["globules", "dormant", "reclaim", "gekko"],
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def expand_semantic_features(tokens: List[str], raw_text: str) -> Counter:
    counts: Counter = Counter()
    for tok in tokens:
        counts[f"tok:{tok}"] += 1.0
        if tok in SEMANTIC_SYNONYMS:
            for syn in SEMANTIC_SYNONYMS[tok]:
                counts[f"syn:{syn}"] += 0.65
    for rev_key, syn_list in SEMANTIC_SYNONYMS.items():
        for syn in syn_list:
            if syn in tokens:
                counts[f"syn:{rev_key}"] += 0.45
    for ng in extract_char_ngrams(raw_text, n=3):
        counts[f"ng:{ng}"] += 0.15
    return counts


def chunk_Words(text: str, source_name: str, chunk_size: int = 120, overlap: int = 25) -> List[Dict[str, Any]]:
    words = text.split()
    chunks: List[Dict[str, Any]] = []
    if not words:
        return chunks
    step = max(1, chunk_size - overlap)
    idx = 0
    chunk_id = 0
    while idx < len(words):
        slice_words = words[idx : idx + chunk_size]
        chunk_text = " ".join(slice_words)
        chunks.append(
            {
                "chunk_id": f"{source_name}::chunk_{chunk_id}",
                "source": source_name,
                "text": chunk_text,
                "tokens": tokenize(chunk_text),
            }
        )
        chunk_id += 1
        if idx + chunk_size >= len(words):
            break
        idx += step
    return chunks


class HybridRAG:
    def __init__(self, docs_dir: str = DOCS_DIR, chunk_size: int = 120, overlap: int = 25):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Dict[str, Any]] = []
        self.doc_freqs: Counter = Counter()
        self.avgdl: float = 1.0
        self.vocab: Dict[str, int] = {}
        self.idf_vector: np.ndarray = np.array([])
        self.doc_matrix: np.ndarray = np.array([])
        self._build_index()

    def _build_index(self) -> None:
        md_files = sorted(glob.glob(os.path.join(self.docs_dir, "*.md")))
        all_chunks: List[Dict[str, Any]] = []
        for path in md_files:
            source_name = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            all_chunks.extend(chunk_Words(content, source_name, self.chunk_size, self.overlap))
        self.chunks = all_chunks
        if not self.chunks:
            return

        total_len = 0
        for ch in self.chunks:
            total_len += len(ch["tokens"])
            for unique_tok in set(ch["tokens"]):
                self.doc_freqs[unique_tok] += 1
        self.avgdl = total_len / max(1, len(self.chunks))

        feature_dfs: Counter = Counter()
        chunk_features: List[Counter] = []
        for ch in self.chunks:
            feats = expand_semantic_features(ch["tokens"], ch["text"])
            chunk_features.append(feats)
            for k in feats.keys():
                feature_dfs[k] += 1

        self.vocab = {feat: i for i, feat in enumerate(sorted(feature_dfs.keys()))}
        num_docs = len(self.chunks)
        num_feats = len(self.vocab)
        self.idf_vector = np.zeros(num_feats, dtype=np.float64)
        for feat, idx in self.vocab.items():
            df = feature_dfs[feat]
            self.idf_vector[idx] = math.log((1.0 + num_docs) / (1.0 + df)) + 1.0

        mat = np.zeros((num_docs, num_feats), dtype=np.float64)
        for d_idx, feats in enumerate(chunk_features):
            for feat, val in feats.items():
                f_idx = self.vocab[feat]
                mat[d_idx, f_idx] = (1.0 + math.log(val)) * self.idf_vector[f_idx]
            norm = np.linalg.norm(mat[d_idx])
            if norm > 0:
                mat[d_idx] /= norm
        self.doc_matrix = mat

    def _bm25_scores(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        n_docs = len(self.chunks)
        scores = np.zeros(n_docs, dtype=np.float64)
        if n_docs == 0 or not query_tokens:
            return scores
        for idx, ch in enumerate(self.chunks):
            dl = len(ch["tokens"])
            tf_counter = Counter(ch["tokens"])
            score = 0.0
            for q_tok in query_tokens:
                df = self.doc_freqs.get(q_tok, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_counter.get(q_tok, 0)
                if tf > 0:
                    denom = tf + k1 * (1.0 - b + b * (dl / max(1e-9, self.avgdl)))
                    score += idf * ((tf * (k1 + 1.0)) / denom)
            scores[idx] = score
        return scores

    def _vector_scores(self, query: str, query_tokens: List[str]) -> np.ndarray:
        n_docs = len(self.chunks)
        if n_docs == 0 or len(self.vocab) == 0:
            return np.zeros(n_docs, dtype=np.float64)
        q_feats = expand_semantic_features(query_tokens, query)
        q_vec = np.zeros(len(self.vocab), dtype=np.float64)
        for feat, val in q_feats.items():
            if feat in self.vocab:
                f_idx = self.vocab[feat]
                q_vec[f_idx] = (1.0 + math.log(val)) * self.idf_vector[f_idx]
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
        return np.dot(self.doc_matrix, q_vec)

    def retrieve_vector_only(self, query: str, top_k: int = 1, min_score: float = 0.05) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        vec_scores = self._vector_scores(query, q_tokens)
        ranked_indices = np.argsort(-vec_scores)
        results: List[Dict[str, Any]] = []
        for idx in ranked_indices[:top_k]:
            score = float(vec_scores[idx])
            if score < min_score:
                continue
            ch = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": ch["chunk_id"],
                    "source": ch["source"],
                    "citation": f"[{ch['source']}]",
                    "score": round(score, 6),
                    "text": ch["text"],
                }
            )
        return results

    def retrieve_hybrid_rrf(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        min_score: float = 0.015,
    ) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        q_tokens = tokenize(query)
        bm25 = self._bm25_scores(q_tokens)
        vec = self._vector_scores(query, q_tokens)

        bm25_order = np.argsort(-bm25)
        vec_order = np.argsort(-vec)

        bm25_ranks = {int(doc_idx): rank + 1 for rank, doc_idx in enumerate(bm25_order)}
        vec_ranks = {int(doc_idx): rank + 1 for rank, doc_idx in enumerate(vec_order)}

        fused_results: List[Dict[str, Any]] = []
        for doc_idx, ch in enumerate(self.chunks):
            has_signal = (bm25[doc_idx] > 0.0) or (vec[doc_idx] > 0.05)
            if not has_signal:
                continue
            r_bm25 = bm25_ranks[doc_idx]
            r_vec = vec_ranks[doc_idx]
            rrf_score = (1.0 / (rrf_k + r_bm25)) + (1.0 / (rrf_k + r_vec))
            if rrf_score >= min_score:
                fused_results.append(
                    {
                        "chunk_id": ch["chunk_id"],
                        "source": ch["source"],
                        "citation": f"[{ch['source']}]",
                        "score": round(float(rrf_score), 6),
                        "bm25_score": round(float(bm25[doc_idx]), 4),
                        "vector_score": round(float(vec[doc_idx]), 4),
                        "text": ch["text"],
                    }
                )

        fused_results.sort(key=lambda x: (x["score"], x["bm25_score"], x["vector_score"]), reverse=True)
        return fused_results[:top_k]


_RAG_INSTANCE: HybridRAG | None = None


def get_rag_engine() -> HybridRAG:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = HybridRAG()
    return _RAG_INSTANCE


def search_patches(query: str, top_k: int = 5, min_score: float = 0.015) -> Dict[str, Any]:
    engine = get_rag_engine()
    hits = engine.retrieve_hybrid_rrf(query=query, top_k=top_k, rrf_k=60, min_score=min_score)
    return {
        "query": query,
        "min_score": min_score,
        "count": len(hits),
        "results": hits,
    }
