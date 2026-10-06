"""Download official revision-pinned Hugging Face models, without tokens or remote code."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from huggingface_hub import snapshot_download
from ragdesk.config import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['all', 'embedding', 'generator'], default='all')
    args = parser.parse_args()
    manifest_path = ROOT/'model-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    selected = ['embedding', 'generator'] if args.model == 'all' else [args.model]
    for key in selected:
        entry = manifest[key]
        local_path = (ROOT/entry['local_path']).resolve()
        snapshot_download(entry['id'], revision=entry['revision'], local_dir=local_path, token=False, allow_patterns=['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'sentencepiece.bpe.model', 'model.safetensors', 'modules.json', 'sentence_bert_config.json', '1_Pooling/config.json'])
        weights = local_path/'model.safetensors'
        with weights.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        expected = entry.get('weights_sha256')
        if expected and expected != digest:
            raise ValueError(f'{key} weights hash mismatch')
        print(f"{key}: {entry['id']} @ {entry['revision']}, SHA256={digest}, license={entry['license']}", flush=True)


if __name__ == '__main__':
    main()
