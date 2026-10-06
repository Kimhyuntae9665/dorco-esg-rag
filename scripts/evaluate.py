"""Page-ground-truth retrieval benchmark; does not score generation correctness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import platform
from datetime import datetime, timezone
from ragdesk.config import ROOT
from ragdesk.retrieval import Retriever
from ragdesk.answering import query_guard


def metrics(rows, k):
    n = len(rows)
    return {'n': n, 'page_hit_at_1': sum(r['first_relevant_rank'] == 1 for r in rows)/n if n else None, f'page_hit_at_{k}': sum(r['first_relevant_rank'] is not None for r in rows)/n if n else None, 'mrr': sum(1/r['first_relevant_rank'] if r['first_relevant_rank'] else 0 for r in rows)/n if n else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT/'data'/'evaluation.json')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10:
        raise ValueError('top-k must be 1..10')
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    retriever = Retriever()
    rows, methods, guard_rows = [], {}, []
    for split, cases in dataset['splits'].items():
        for case in cases:
            if not case['answerable']:
                guard_rows.append({'id': case['id'], 'split': split, 'language': case['language'], 'guard': query_guard(case['question']), 'guard_fired': query_guard(case['question']) is not None})
                continue
            expected = set(case['expected_pages'])
            for method in ('bm25', 'dense', 'hybrid'):
                result = retriever.search(case['question'], method, args.top_k)
                hit_rank = next((i for i, e in enumerate(result['evidence'], 1) if e['doc_id'] == case['expected_doc_id'] and e['page'] in expected), None)
                rows.append({'id': case['id'], 'split': split, 'language': case['language'], 'method': method, 'expected_doc_id': case['expected_doc_id'], 'expected_pages': sorted(expected), 'first_relevant_rank': hit_rank, 'elapsed_ms': result['elapsed_ms'], 'retrieved': [{'rank': i, 'chunk_id': e['chunk_id'], 'doc_id': e['doc_id'], 'page': e['page'], 'score': e['score']} for i, e in enumerate(result['evidence'], 1)]})
        for method in ('bm25', 'dense', 'hybrid'):
            subset = [r for r in rows if r['split'] == split and r['method'] == method]
            methods.setdefault(method, {})[split] = {**metrics(subset, args.top_k), 'by_language': {lang: metrics([r for r in subset if r['language'] == lang], args.top_k) for lang in sorted({r['language'] for r in subset})}}
    output = {'ready': True, 'generated_at': datetime.now(timezone.utc).isoformat(), 'scope': dataset['scope'], 'dataset_version': dataset['version'], 'configuration': {'top_k': args.top_k, 'metric_definition': 'mrr_at_k, no top-k hit scores zero', 'same_corpus': True, 'page_deduplication': True, 'rrf_constant': 60, 'rrf_candidates': 100, 'dense_similarity': 'normalized_cosine', 'embedding': json.loads((ROOT/'model-manifest.json').read_text())['embedding'], 'dev_calibration': 'No parameter tuning performed', 'seed': 0, 'deterministic_rank_ties': 'stable corpus order', 'python': platform.python_version(), 'generation_scored': False}, 'methods': methods, 'results': rows, 'unanswerable_guard_cases': guard_rows, 'limitations': ['Small author-created benchmark; not independent supplier validation', 'Page-level relevance labels; no automatic answer entailment score', 'mrr field is MRR@k; relevant pages below retrieval cutoff receive zero', 'Query guard coverage is reported separately from retrieval metrics']}
    (ROOT/'docs').mkdir(exist_ok=True)
    (ROOT/'docs'/'evaluation-results.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(methods, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
