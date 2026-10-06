"""Live HTTP boundary checks; start server.py before running."""
import json
import urllib.request
import urllib.error


def fetch(path, expected, body=None, headers=None):
    merged = {'Content-Type': 'application/json', **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request('http://127.0.0.1:8770'+path, data=data, headers=merged)
    try:
        response = urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as error:
        response = error
    assert response.code == expected, (path, expected, response.code)
    return response.read()


fetch('/api/status', 200)
fetch('/api/status', 403, headers={'Host': 'attacker.example'})
fetch('/api/status', 403, headers={'Origin': 'https://attacker.example'})
fetch('/../model-manifest.json', 404)
fetch('/api/page?doc_id=../secrets&page=1', 400)
fetch('/api/page?doc_id=kesg_supply&page=99999', 400)
assert fetch('/api/page?doc_id=kesg_supply&page=36', 200).startswith(b'\x89PNG')
fetch('/api/search', 400, {'question': 'x', 'method': 'invented'})
fetch('/api/search', 400, {'question': 'x', 'top_k': True})
fetch('/api/search', 400, {'question': 'x'*1501})
fetch('/api/search', 400, {'question': 'x'*17000})
fetch('/api/answer', 400, {'question': 'x', 'language': 'fr'})
fetch('/api/search', 400, {'question': 'x', 'doc_id': '../private'})
filtered = json.loads(fetch('/api/search', 200, {'question': '에너지 관리', 'method': 'hybrid', 'doc_id': 'dorco_profile'}))
assert all(e['doc_id'] == 'dorco_profile' for e in filtered['evidence'])
guarded = json.loads(fetch('/api/answer', 200, {'question': '도루코 실제 협력사별 ESG 점수 알려줘', 'language': 'ko'}))
assert guarded['status'] == 'abstained' and not guarded['citations']
print('15 live HTTP boundary checks passed')
