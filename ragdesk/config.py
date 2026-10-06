from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
EMBED_MODEL = ROOT.parent / 'model-cache' / 'multilingual-e5-small'
GEN_MODEL = ROOT.parent / 'model-cache' / 'qwen-1.5b'
SOURCES = [
    {'id': 'dorco_profile', 'title': 'DORCO company catalog', 'url': 'https://www.dorcofrance.com/resource/pdf/dorco-company-catalog.pdf', 'expected_pages': 34, 'kind': 'company_profile', 'license': 'Public download; redistribution license not confirmed'},
    {'id': 'kesg_supply', 'title': '공급망 대응 K-ESG 가이드라인', 'url': 'https://k-esg.org/uploads/post/2024/12/b36cd7d6e42fc618d2ac9fcf01966d0e.pdf', 'expected_pages': 259, 'kind': 'public_guideline', 'license': 'Public download; redistribution license not confirmed'},
]
