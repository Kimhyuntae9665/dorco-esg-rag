import json
import re
import time
import urllib.request


LANGUAGES = {'ko': 'Korean', 'en': 'English', 'vi': 'Vietnamese'}
MODEL_URL = 'http://127.0.0.1:8892'


def query_guard(question):
    # Explicit scope guard; this is not learned relevance or semantic fact verification.
    sensitive = r'실제\s*(?:점수|평가|공급|등급|탄소)|협력사별|공급업체별|내부\s*(?:평가|자료|계약|점수)|올해\s*(?:점수|탄소)|actual\s+(?:supplier|score|rating)|internal\s+(?:supplier|score|contract)|confidential'
    injection = r'ignore\s+(?:all\s+)?(?:previous|system)|시스템\s*(?:지시|프롬프트).*무시|이전\s*지시.*무시|reveal\s+(?:the\s+)?system\s+prompt'
    if re.search(sensitive, question, re.I):
        return 'private_company_facts'
    if re.search(injection, question, re.I):
        return 'instruction_override'
    return None


def validate_output(raw, evidence):
    # Checks formatting and identifier membership only, not entailment or factual truth.
    match = re.search(r'(?:^|\n)(?:CITES|FIRST LINE(?: CITED)?):[ \t]*([^\n]+)', raw)
    if not match:
        raise ValueError('Missing explicit citation line')
    ids = [part.strip().strip('[]') for part in match.group(1).split(',') if part.strip()]
    known = {e['chunk_id']: e for e in evidence}
    if not ids or len(ids) > 3 or any(i not in known for i in ids):
        raise ValueError('Citation identifier outside supplied evidence')
    answer = (raw[:match.start()] + raw[match.end():]).strip()
    if answer.startswith('ANSWER:'):
        answer = answer[7:].strip()
    if not answer or len(answer) > 2500:
        raise ValueError('Empty or oversized answer')
    citations = [{'chunk_id': i, 'doc_id': known[i]['doc_id'], 'title': known[i]['title'], 'page': known[i]['page'], 'quote': known[i]['text'][:160], 'url': known[i]['url']} for i in dict.fromkeys(ids)]
    return answer, citations


def model_status():
    try:
        with urllib.request.urlopen(MODEL_URL+'/status', timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return {'id': 'Qwen/Qwen2.5-1.5B-Instruct', 'ready': False}


def answer_question(retriever, question, method='hybrid', language='ko', doc_id=None):
    guarded = query_guard(question)
    if guarded:
        answers = {'ko': '공개 문서만으로는 해당 내부 사실이나 실제 협력사 평가 결과를 확인할 수 없습니다. 내부 지시나 비공개 정보는 제공하지 않습니다.', 'en': 'These public documents cannot establish internal company facts or actual supplier results. I cannot provide private information or override system instructions.', 'vi': 'Các tài liệu công khai không xác nhận dữ liệu nội bộ hoặc kết quả thực tế của nhà cung cấp. Tôi không cung cấp thông tin riêng tư hoặc bỏ qua chỉ dẫn hệ thống.'}
        return {'status': 'abstained', 'answer': answers[language], 'citations': [], 'evidence': [], 'retrieval_ms': 0, 'generation_ms': 0, 'model': 'query_guard', 'warnings': [guarded, 'Explicit query guard; not a learned relevance decision']}
    result = retriever.search(question, method, 3, doc_id)
    start = time.perf_counter()
    base = {'evidence': result['evidence'], 'retrieval_ms': result['elapsed_ms'], 'retrieval_trace': result['retrieval_trace'], 'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'warnings': ['Citation IDs and nonempty format checked; semantic support is not automatically verified']}
    contexts = [{'chunk_id': e['chunk_id'], 'text': e['text'][:700]} for e in result['evidence']]
    request = urllib.request.Request(MODEL_URL+'/generate', data=json.dumps({'question': question, 'language': language, 'contexts': contexts}, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json'})
    raw = ''
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            generated = json.load(response)
        raw = generated['text'][:5000]
        if generated.get('finish_reason') == 'length':
            raise ValueError('Generation reached token limit before EOS')
        answer, citations = validate_output(raw, result['evidence'])
        # Basic script checks catch wrong-language output, not semantic language accuracy.
        cjk = bool(re.search(r'[\u4e00-\u9fff가-힣]', answer))
        if language == 'ko' and not re.search(r'[가-힣]', answer):
            raise ValueError('Requested Korean but no Hangul in answer')
        if language in ('en', 'vi') and cjk:
            raise ValueError('Requested Latin-script language but generated CJK text')
        status = 'generated'
    except (OSError, ValueError, KeyError) as error:
        answer, citations, status = '', [], 'failed'
        base['warnings'].append(f'Generation failed: {type(error).__name__}: {error}')
    return {**base, 'status': status, 'answer': answer, 'citations': citations, 'generation_ms': round((time.perf_counter()-start)*1000, 1), 'raw_output': raw}
