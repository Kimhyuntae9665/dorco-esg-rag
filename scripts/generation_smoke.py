"""Real local generation calls. Full transient context stays in ignored runtime/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import urllib.request
from ragdesk.config import RUNTIME

cases = [('ko', '화학물질 관리 점검을 준비할 때 어떤 대장과 조사서를 모아야 하나요?'), ('en', 'What environmental initiatives does DORCO describe in its ESG catalog?'), ('vi', 'Theo hướng dẫn K-ESG, cần thu thập hồ sơ nào để kiểm tra quản lý hóa chất?')]
results = []
for language, question in cases:
    request = urllib.request.Request('http://127.0.0.1:8770/api/answer', data=json.dumps({'question': question, 'language': language, 'method': 'dense'}).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except OSError as error:
        result = {'status': 'failed', 'error': str(error)}
    results.append({'language': language, 'question': question, **result})
    (RUNTIME/'generation-smoke.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(language, result['status'], result.get('generation_ms'), result.get('answer', ''), result.get('warnings', []), flush=True)
print('Saved ignored runtime/generation-smoke.json. Review semantic support manually.', flush=True)
