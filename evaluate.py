"""Manifest loading, model execution and result export.

The original scoring functions remain in pronunciation.py, text_methods.py,
similarity.py and naturalness.py. Only one model occupies GPU memory at a time.
"""
import csv
import gc
import hashlib
import json
import math
from pathlib import Path

import audio
import pronunciation
import text_methods
import similarity
import naturalness

CRITERIA = ('pronunciation', 'naturalness', 'speaker_similarity')
FIELDS = ('clip_id', 'audio_path', 'language', 'text',
          'reference_audio_path', 'target_word', 'contrast')


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('The manifest is missing required columns.')
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row['clip_id'] or row['clip_id'] in seen:
            raise ValueError('Empty or duplicate clip ID.')
        seen.add(row['clip_id'])
        row['language'] = {'Hindi': 'hi', 'Tamil': 'ta'}.get(row['language'], row['language'])
        if row['language'] not in ('hi', 'ta') or not row['text'].strip():
            raise ValueError('Each clip needs a language and intended text.')
        text_methods.predict_word('', row['target_word'], row['contrast'])
        for key in ('audio_path', 'reference_audio_path'):
            if row[key]:
                row[key] = str((path.parent / row[key]).resolve())
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
            elif key == 'audio_path':
                raise ValueError('Missing audio path.')
        if row['audio_path'] == row['reference_audio_path']:
            raise ValueError('A clip cannot be its own speaker reference.')
    return sorted(rows, key=lambda row: row['clip_id'])


def cache_key(row, criterion, model_fingerprint):
    """Recompute a result when the relevant audio, text, code or model changes."""
    data = {'audio': file_hash(row['audio_path']), 'model': model_fingerprint}
    modules = ['audio.py', 'schema.py', 'evaluate.py']
    if criterion == 'pronunciation':
        data.update(text=row['text'], language=row['language'],
                    target=row['target_word'], contrast=row['contrast'])
        modules += ['pronunciation.py', 'text_methods.py']
    elif criterion == 'naturalness':
        modules += ['naturalness.py']
    else:
        data['reference'] = file_hash(row['reference_audio_path'])
        modules += ['similarity.py']
    data['source'] = {name: file_hash(Path(__file__).parent / name) for name in modules}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_model(criterion, device, versions):
    import torch
    if criterion == 'pronunciation':
        import whisper
        return whisper.load_model('small', device=device), None
    if criterion == 'naturalness':
        source = Path(__file__).parent / 'outputs/reproducibility/speechmos_source'
        model = torch.hub.load(str(source), 'utmos22_strong', source='local', trust_repo=True)
        return model.to(device).eval(), None
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    name = 'microsoft/wavlm-base-plus-sv'
    revision = versions['weights']['wavlm_revision']
    processor = Wav2Vec2FeatureExtractor.from_pretrained(name, revision=revision)
    model = WavLMForXVector.from_pretrained(name, revision=revision)
    return model.to(device).eval(), processor


def evaluate_manifest(manifest_path, output, require_complete=False):
    import torch
    rows = load_manifest(manifest_path)
    missing = [row['clip_id'] for row in rows if not row['reference_audio_path']]
    if missing and require_complete:
        raise ValueError(f'{len(missing)} clips still need a confirmed speaker reference.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    version_path = Path(__file__).parent / 'outputs/reproducibility/model_versions.json'
    versions = json.loads(version_path.read_text())
    # Verify the recorded weight files before trusting any saved predictions.
    weights = versions['weights']
    paths = {weights['whisper_small']['path']: weights['whisper_small']['sha256']}
    paths.update(weights['speechmos_cached_files'])
    paths.update(weights['wavlm_files'])
    for path, expected in paths.items():
        if file_hash(path) != expected:
            raise ValueError(f'Model weights changed: {path}')
    model_fingerprint = hashlib.sha256(version_path.read_bytes()).hexdigest()
    source = Path(__file__).parent / 'outputs/reproducibility/speechmos_source'
    source_fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source.rglob('*.py')))).hexdigest()
    model_fingerprint += source_fingerprint
    cache_path = output / 'cache.json'
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    reused = 0
    computed = 0
    for criterion in CRITERIA:
        work = []
        for row in rows:
            if criterion == 'speaker_similarity' and not row['reference_audio_path']:
                continue
            key = cache_key(row, criterion, model_fingerprint)
            entry = cache.get(row['clip_id'], {}).get(criterion)
            if entry and entry['key'] == key:
                reused += 1
            else:
                work.append((row, key))
        if not work:
            print(f'{criterion}: using verified saved results', flush=True)
            continue
        model, processor = load_model(criterion, device, versions)
        reference_embeddings = {}
        for index, (row, key) in enumerate(work, 1):
            waveform, rate = audio.prepare_audio(row['audio_path'])
            result = {'key': key}
            if criterion == 'pronunciation':
                transcript = pronunciation.transcribe_audio(model, waveform, rate, row['language'])
                details = text_methods.score_pronunciation(row['text'], transcript)
                result.update(score=details['score'], transcript=transcript, details=details,
                              predicted_word=text_methods.predict_word(transcript, row['target_word'], row['contrast']))
            elif criterion == 'naturalness':
                result['score'] = naturalness.score_naturalness(model, waveform, rate)
            else:
                reference = row['reference_audio_path']
                if reference not in reference_embeddings:
                    ref_waveform, ref_rate = audio.prepare_audio(reference)
                    reference_embeddings[reference] = similarity.extract_speaker_embedding(model, processor, ref_waveform, ref_rate)
                embedding = similarity.extract_speaker_embedding(model, processor, waveform, rate)
                result['score'] = similarity.cosine_compare_speakers(embedding, reference_embeddings[reference])
            if not math.isfinite(result['score']):
                raise ValueError('Nonfinite score.')
            cache.setdefault(row['clip_id'], {})[criterion] = result
            temporary = cache_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(cache_path)
            computed += 1
            print(f'{criterion}: {index}/{len(work)}', flush=True)
        del model, processor, reference_embeddings
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    scores = []
    details = []
    for row in rows:
        record = cache[row['clip_id']]
        details.append({'clip_id': row['clip_id'], 'text': row['text'],
                        'transcript': record['pronunciation']['transcript'],
                        'predicted_word': record['pronunciation']['predicted_word']})
        for criterion in CRITERIA:
            if criterion == 'speaker_similarity' and not row['reference_audio_path']:
                continue
            scores.append({'clip_id': row['clip_id'], 'criterion': criterion,
                           'score': record[criterion]['score'],
                           'predicted_word': record['pronunciation']['predicted_word']})
    with (output / 'scores.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['clip_id', 'criterion', 'score', 'predicted_word'])
        writer.writeheader()
        writer.writerows(scores)
    (output / 'transcripts.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'complete': not missing, 'clips': len(rows), 'score_rows': len(scores),
               'missing_speaker_similarity': missing, 'reused_scores': reused, 'new_scores': computed,
               'human_agreement_metrics': 'Not measured: independent listener answers are required.'}
    (output / 'status.json').write_text(json.dumps(summary, indent=2))
    print(f"Saved {len(scores)} score rows to {output / 'scores.csv'}", flush=True)
    if missing:
        print(f'DEVELOPMENT OUTPUT: {len(missing)} speaker scores are missing. See status.json.', flush=True)
