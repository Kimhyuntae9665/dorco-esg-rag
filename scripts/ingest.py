"""Download fixed public PDFs, extract page chunks, and build learned cosine vectors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import re
import urllib.request
import fitz
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from ragdesk.config import ROOT, RUNTIME, EMBED_MODEL, SOURCES


def page_chunks(text, limit=900, overlap=180):
    # K-ESG PDF encodes word spaces as private-use glyph U+F613.
    text = text.replace('\uf613', ' ')
    text = re.sub(r'[ \t]+', ' ', text).strip()
    start = 0
    while start < len(text):
        end = min(len(text), start+limit)
        if end < len(text):
            boundary = max(text.rfind('\n', start+limit//2, end), text.rfind(' ', start+limit//2, end))
            if boundary > start:
                end = boundary
        yield start, end, text[start:end].strip()
        if end == len(text):
            break
        start = max(start+1, end-overlap)


def main():
    (RUNTIME/'pdfs').mkdir(parents=True, exist_ok=True)
    chunks, docs = [], []
    for source in SOURCES:
        path = RUNTIME/'pdfs'/f"{source['id']}.pdf"
        if not path.exists():
            urllib.request.urlretrieve(source['url'], path)
        pdf = fitz.open(path)
        if len(pdf) != source['expected_pages']:
            raise ValueError(f"{source['id']}: expected {source['expected_pages']} pages, got {len(pdf)}")
        pages = []
        start_count = len(chunks)
        for number, page in enumerate(pdf, 1):
            text = page.get_text('text', sort=True)
            pages.append({'page': number, 'text': text})
            for sequence, (start, end, excerpt) in enumerate(page_chunks(text), 1):
                if len(excerpt) < 25:
                    continue
                chunks.append({'chunk_id': f"{source['id']}:p{number:03d}:c{sequence:02d}", 'doc_id': source['id'], 'title': source['title'], 'page': number, 'text': excerpt, 'url': source['url'], 'kind': source['kind'], 'start_char': start, 'end_char': end})
        (RUNTIME/f"{source['id']}-pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding='utf-8')
        docs.append({**source, 'pages': len(pdf), 'chunks': len(chunks)-start_count, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Extracted {source['id']}: {len(pdf)} pages, {len(chunks)-start_count} chunks", flush=True)
    (RUNTIME/'chunks.jsonl').write_text('\n'.join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding='utf-8')
    torch.set_num_threads(6)
    model = SentenceTransformer(str(EMBED_MODEL), device='cpu', local_files_only=True)
    token_lengths = [len(model.tokenizer('passage: '+c['text'])['input_ids']) for c in chunks]
    vectors = model.encode(['passage: '+c['text'] for c in chunks], normalize_embeddings=True, batch_size=24, convert_to_numpy=True, show_progress_bar=True)
    np.save(RUNTIME/'embeddings.npy', vectors.astype(np.float32), allow_pickle=False)
    manifest = {'documents': docs, 'total_pages': sum(d['pages'] for d in docs), 'total_chunks': len(chunks), 'chunking': {'unit': 'page', 'max_characters': 900, 'overlap_characters': 180, 'normalization': 'U+F613 PDF space glyph replaced by space; tabs/spaces collapsed'}, 'embedding_dimension': int(vectors.shape[1]), 'vector_dtype': 'float32', 'normalized': True, 'embedding_token_limit': model.max_seq_length, 'chunks_above_embedding_token_limit': sum(n > model.max_seq_length for n in token_lengths), 'maximum_chunk_tokens': max(token_lengths)}
    (RUNTIME/'corpus-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Public manifest contains metadata only, never extracted text or embeddings.
    (ROOT/'sources-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Ready: {len(chunks)} x {vectors.shape[1]} real learned embeddings", flush=True)


if __name__ == '__main__':
    main()
