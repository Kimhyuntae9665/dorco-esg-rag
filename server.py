import json
import mimetypes
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
import fitz
from ragdesk.config import ROOT, RUNTIME, SOURCES
from ragdesk.retrieval import Retriever
from ragdesk.answering import answer_question, model_status, LANGUAGES
from ragdesk.http_utils import LocalHandler, confined_file

retriever = None
load_error = None


class Handler(LocalHandler):
    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        try:
            if path == '/api/status':
                models = json.loads((ROOT/'model-manifest.json').read_text(encoding='utf-8'))
                evaluation = ROOT/'docs'/'evaluation-results.json'
                self.send_json({'ready': retriever is not None, 'error': load_error, 'corpus': retriever.manifest if retriever else None, 'embedding': models['embedding'], 'generator': model_status(), 'retrievalmethods': ['bm25', 'dense', 'hybrid'], 'assessment': json.loads(evaluation.read_text(encoding='utf-8')) if evaluation.exists() else None})
            elif path == '/api/evaluation':
                evaluation = ROOT/'docs'/'evaluation-results.json'
                self.send_json(json.loads(evaluation.read_text(encoding='utf-8')) if evaluation.exists() else {'ready': False, 'status': 'not_run'})
            elif path == '/api/page':
                query = parse_qs(urlsplit(self.path).query)
                doc_id = query.get('doc_id', [''])[0]
                page = int(query.get('page', ['0'])[0])
                source = next((s for s in SOURCES if s['id'] == doc_id), None)
                if source is None or not 1 <= page <= source['expected_pages']:
                    raise ValueError('Unknown manifest document or page')
                with fitz.open(RUNTIME/'pdfs'/f'{doc_id}.pdf') as pdf:
                    self.send_bytes(pdf[page-1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).tobytes('png'), 'image/png')
            else:
                file = confined_file(ROOT/'web', self.path)
                if file:
                    self.send_bytes(file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                else:
                    self.send_json({'error': 'Not found'}, 404)
        except (ValueError, OSError) as error:
            self.send_json({'error': str(error)}, 400)

    def do_POST(self):
        if not self.allowed():
            return
        try:
            if self.path not in ('/api/search', '/api/answer'):
                self.send_json({'error': 'Not found'}, 404)
                return
            body = self.body()
            question, method = body.get('question'), body.get('method', 'hybrid')
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
                raise ValueError('Question length must be 1..1500 characters')
            if method not in ('bm25', 'dense', 'hybrid'):
                raise ValueError('Unknown retrieval method')
            doc_id = body.get('doc_id')
            if doc_id == 'all':
                doc_id = None
            if doc_id not in (None, *[s['id'] for s in SOURCES]):
                raise ValueError('Unknown document filter')
            if retriever is None:
                self.send_json({'error': 'Index is not ready', 'detail': load_error}, 503)
                return
            if self.path == '/api/search':
                top_k = body.get('top_k', 3)
                if type(top_k) is not int or not 1 <= top_k <= 10:
                    raise ValueError('top_k must be an integer 1..10')
                self.send_json(retriever.search(question.strip(), method, top_k, doc_id))
            else:
                language = body.get('language', 'ko')
                if not isinstance(language, str) or language not in LANGUAGES:
                    raise ValueError('Language must be ko, en, or vi')
                self.send_json(answer_question(retriever, question.strip(), method, language, doc_id))
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({'error': str(error)}, 400)
        except Exception as error:
            self.send_json({'error': f'Request failed: {type(error).__name__}'}, 500)


if __name__ == '__main__':
    try:
        retriever = Retriever()
    except Exception as error:
        load_error = f'{type(error).__name__}: {error}'
    print(f'RAG API http://127.0.0.1:8770 ready={retriever is not None}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8770), Handler).serve_forever()
