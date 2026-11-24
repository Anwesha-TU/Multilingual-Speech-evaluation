import unicodedata

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text= text.lower()

    chars = []

    for character in text:
        if unicodedata.category(character).startswith("P"):
            chars.append(" ")
        else:
            chars.append(character)
    text = "".join(chars)

    return " ".join(text.split())

def edit_distance(expected, recognized):
    rows = len(expected) + 1
    columns = len(recognized) + 1

    table = [
        [0] * columns
        for _ in range(rows)
    ]


    for i in range(rows):
        table[i][0] = i

    
    for j in range(columns):
        table[0][j] = j

    for i in range(1, rows):
        for j in range(1, columns):
            if expected[i - 1] == recognized[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            deletion = table[i - 1][j] + 1
            insertion = table[i][j - 1] + 1
            substitution = (
                table[i - 1][j - 1] + substitution_cost
            )

            table[i][j] = min(
                deletion,
                insertion,
                substitution,
            )

    return table[-1][-1]


def score_pronunciation(expected_text, recognized_text):
    expected_text = normalize_text(expected_text)
    recognized_text = normalize_text(recognized_text)

    if not expected_text:
        raise ValueError("Expected text is empty after normalization")

    edits = edit_distance(expected_text, recognized_text)

    exp_chars = len(expected_text)

    cer = edits / exp_chars
    score = 1.0 - cer

    return {
        "expected_normalized": expected_text,
        "recognized_normalized": recognized_text,
        "edit_count": edits,
        "expected_character_count": exp_chars,
        "cer": cer,
        "score": score,

    } 

def predict_word(transcript, target, contrast):
    """Return an actual candidate word, or unclear; never use intended text as truth."""
    if not target and not contrast:
        return ''
    if not target or not contrast or normalize_text(target) == normalize_text(contrast):
        raise ValueError('Two distinct candidate words are required.')
    tokens = set(normalize_text(transcript).split())
    target_found = normalize_text(target) in tokens
    contrast_found = normalize_text(contrast) in tokens
    if target_found and not contrast_found:
        return target
    if contrast_found and not target_found:
        return contrast
    return 'unclear'


def write_confusion_matrices(manifest_path, scores_path, heard_path, output):
    """Optional diagnostic; page 20 requires accuracy against agreed human labels."""
    import csv
    import json
    from collections import Counter, defaultdict
    from pathlib import Path

    if not Path(heard_path).is_file():
        print('No confusion matrix created: answers/heard_words.csv is missing.')
        print('It needs two independent listeners\' actual heard-word labels. Intended text is not ground truth.')
        return

    def read(path):
        with Path(path).open(encoding='utf-8-sig', newline='') as file:
            return list(csv.DictReader(file))

    candidates = {r['clip_id']: r for r in read(manifest_path) if r['target_word']}
    predictions = {}
    for row in read(scores_path):
        if row['criterion'] == 'pronunciation' and row['clip_id'] in candidates:
            if row['clip_id'] in predictions or not row['predicted_word']:
                raise ValueError('Duplicate or missing word prediction.')
            predictions[row['clip_id']] = row['predicted_word']
    if set(predictions) != set(candidates):
        raise ValueError('Missing minimal-pair predictions.')
    labels = defaultdict(dict)
    for row in read(heard_path):
        cid, rater, label = row['clip_id'], row['rater_id'], row['heard_word']
        if cid not in candidates or not rater or rater in labels[cid]:
            raise ValueError('Unknown clip or duplicate/empty rater.')
        c = candidates[cid]
        if label not in (c['target_word'], c['contrast'], 'unclear'):
            raise ValueError(f'Invalid heard-word label for {cid}. Store the actual word.')
        labels[cid][rater] = label
    if set(labels) != set(candidates) or any(len(v) != 2 for v in labels.values()):
        raise ValueError('Each minimal-pair clip requires two independent heard-word labels.')

    tables = defaultdict(Counter)
    counts = defaultdict(Counter)
    for cid, row in candidates.items():
        language = {'Hindi': 'hi', 'Tamil': 'ta'}.get(row['language'], row['language'])
        a, b = labels[cid].values()
        if a != b:
            counts[language]['disagreements_excluded'] += 1
            continue

        def category(value):
            if value == row['target_word']: return 'target'
            if value == row['contrast']: return 'contrast'
            if value == 'unclear': return 'unclear'
            return 'invalid'

        tables[language][category(a), category(predictions[cid])] += 1
        counts[language]['agreed_items'] += 1
        counts[language]['correct'] += int(a == predictions[cid])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for language in ('hi', 'ta'):
        count = counts[language]
        n = count['agreed_items']
        report[language] = dict(count, accuracy=count['correct'] / n if n else None,
                                qualifies_for_official_measure=n >= 10)
        with (output / f'confusion_{language}.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['human_label', 'predicted_target', 'predicted_contrast', 'predicted_unclear', 'predicted_invalid'])
            for truth in ('target', 'contrast', 'unclear'):
                writer.writerow([truth] + [tables[language][truth, pred] for pred in ('target', 'contrast', 'unclear', 'invalid')])
    (output / 'word_accuracy.json').write_text(json.dumps(report, indent=2))
    print('Saved confusion tables and word accuracy from agreed human labels.')
