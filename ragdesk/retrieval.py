import json
import math
import re
import time
from collections import Counter
import numpy as np
from .config import RUNTIME, EMBED_MODEL, SOURCES


def tokenize(text):
    words = re.findall(r'[a-z0-9]+|[가-힣]+|[\u00c0-\u024f]+', text.lower())
    # Korean compound nouns have no spaces. Character bigrams supplement lexical BM25.
    return words + [w[i:i+2] for w in words if re.fullmatch(r'[가-힣]+', w) for i in range(len(w)-1)]


class BM25:
    def __init__(self, texts, k1=1.5, b=.75):
        self.counts = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avg = max(1, float(self.lengths.mean()))
        df = Counter(t for c in self.counts for t in c)
        self.idf = {t: math.log(1 + (len(texts)-n+.5)/(n+.5)) for t, n in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        scores = np.zeros(len(self.counts))
        for token in set(tokenize(query)):
            freq = np.array([c.get(token, 0) for c in self.counts])
            scores += self.idf.get(token, 0) * freq * (self.k1+1) / (freq + self.k1*(1-self.b+self.b*self.lengths/self.avg))
        return scores


def rrf(rankings, n, constant=60):
    scores = np.zeros(n)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (constant + rank)
    return scores


class Retriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(6)
        self.chunks = [json.loads(line) for line in (RUNTIME/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        self.vectors = np.load(RUNTIME/'embeddings.npy', allow_pickle=False)
        if len(self.chunks) != len(self.vectors):
            raise ValueError('Chunk/vector count mismatch')
        self.model = SentenceTransformer(str(EMBED_MODEL), local_files_only=True, device='cpu')
        self.bm25 = BM25([c['text'] for c in self.chunks])
        self.manifest = json.loads((RUNTIME/'corpus-manifest.json').read_text(encoding='utf-8'))

    def search(self, question, method='hybrid', top_k=3, doc_id=None):
        start = time.perf_counter()
        if method not in ('hybrid', 'dense', 'bm25'):
            raise ValueError('Unknown retrieval method')
        if doc_id not in (None, *[s['id'] for s in SOURCES]):
            raise ValueError('Unknown document filter')
        candidates = np.array([i for i, c in enumerate(self.chunks) if doc_id is None or c['doc_id'] == doc_id])
        trace = []
        dense = lexical = None
        if method in ('hybrid', 'dense'):
            vector = self.model.encode(['query: '+question], normalize_embeddings=True, convert_to_numpy=True)[0]
            dense = self.vectors @ vector
            trace.extend(['multilingual_e5_query_embedding', 'normalized_numpy_cosine'])
        if method in ('hybrid', 'bm25'):
            lexical = self.bm25.scores(question)
            trace.append('bm25_korean_bigrams')
        if method == 'hybrid':
            dense_rank = candidates[np.argsort(-dense[candidates], kind='stable')[:100]]
            lexical_rank = [i for i in candidates[np.argsort(-lexical[candidates], kind='stable')[:100]] if lexical[i] > 0]
            scores = rrf([dense_rank, lexical_rank], len(self.chunks))
            trace.append('reciprocal_rank_fusion_k60_top100')
        else:
            scores = dense if method == 'dense' else lexical
        ranked = candidates[np.argsort(-scores[candidates], kind='stable')]
        # Keep distinct page evidence so duplicate overlapping chunks do not fill all slots.
        chosen, pages = [], set()
        for i in ranked:
            c = self.chunks[int(i)]
            key = (c['doc_id'], c['page'])
            if key not in pages:
                chosen.append({**c, 'score': round(float(scores[i]), 6)})
                pages.add(key)
            if len(chosen) == top_k:
                break
        if doc_id:
            trace.insert(0, 'explicit_document_filter:'+doc_id)
        return {'question': question, 'method': method, 'doc_id': doc_id, 'elapsed_ms': round((time.perf_counter()-start)*1000, 1), 'evidence': chosen, 'retrieval_trace': trace}
