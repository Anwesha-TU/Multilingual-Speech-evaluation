"""Reproduce development metrics from genuine exported human reviews.

Run: python evaluate_humans.py
No model inference, imputation, calibration, or fabricated listener identities.
Coverage sets are NOT people. Pairwise preferences require the same real raters.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import csv
import hashlib
import json
import math
import unicodedata
import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
CRITERIA = ('naturalness', 'pronunciation', 'speaker_similarity')
OUT = ROOT / 'outputs/evaluation'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def nfc(value):
    return unicodedata.normalize('NFC', value.strip())


def rho(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    return float(spearmanr(x, y).statistic)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = read(ROOT / 'inputs/manifest.csv')
    manifest = {r['clip_id']: r for r in manifest_rows}
    assert len(manifest) == len(manifest_rows)
    metadata = {r['clip_id']: r for r in read(ROOT / 'answers/clip_metadata.csv')}
    score_rows = read(ROOT / 'outputs/current/scores.csv')
    scores = {(r['clip_id'], r['criterion']): r for r in score_rows}
    assert len(scores) == len(score_rows) == 3 * len(manifest)
    assert set(scores) == {(cid, c) for cid in manifest for c in CRITERIA}
    for r in scores.values():
        r['score'] = float(r['score'])
        assert math.isfinite(r['score'])

    reviews = defaultdict(dict)
    clean, changes, coverage, rejected = [], [], [], []
    for name in ['hi_set_1.csv', 'hi_set_2.csv', 'ta_set_1.csv', 'ta_set_2_completed.csv']:
        source = ROOT / 'answers/raw' / name
        rows = read(source)
        expected_language, _, expected_set = source.stem.split('_')[:3]
        ids = set()
        accepted_raters = set()
        for line, raw in enumerate(rows, 2):
            row = {k: nfc(v or '') for k, v in raw.items()}
            if row != raw:
                changes.append(dict(file=source.name, line=line, action='Trim surrounding whitespace / Unicode NFC only'))
            cid, rat = row['clip_id'], row['rater_id']
            provenance = row.pop('rating_source', '')
            if provenance not in ('', 'original_set_1', 'original_set_2') or not rat:
                rejected.append(dict(source_file=source.name, line=line, clip_id=cid,
                    rating_source=provenance, reason='Not an independent identified human review',
                    raw_row_json=json.dumps(raw, ensure_ascii=False)))
                continue
            accepted_raters.add(rat)
            assert cid in manifest and rat, (source, line, 'unknown clip or empty rater')
            assert row['language'] == manifest[cid]['language'] == expected_language
            assert row['coverage_set'] == expected_set and cid not in ids
            ids.add(cid)
            for c in CRITERIA:
                value = float(row[c])
                assert math.isfinite(value) and value.is_integer() and 1 <= value <= 5
                row[c] = int(value)
            m = manifest[cid]
            if m['target_word']:
                assert row['heard_word'] in {nfc(m['target_word']), nfc(m['contrast']), 'unclear'}, (cid, row['heard_word'])
            else:
                assert not row['heard_word'], (cid, 'unexpected heard word')
            assert rat not in reviews[cid], (cid, 'duplicate listener across exports')
            reviews[cid][rat] = row
            clean.append(dict(row, source_file=source.name))
        coverage.append(dict(file=source.name, reviews=len(ids), distinct_clips=len(ids), missing_clips=120-len(ids), contributors=len(accepted_raters)))
    fields = ['coverage_set','language','clip_id','rater_id',*CRITERIA,'heard_word','updated_at','source_file']
    write(ROOT/'answers/reviews_clean.csv', clean, fields)
    ratings = [dict(clip_id=r['clip_id'],criterion=c,rater_id=r['rater_id'],rating=r[c]) for r in clean for c in CRITERIA]
    words = [dict(clip_id=r['clip_id'],rater_id=r['rater_id'],heard_word=r['heard_word']) for r in clean if manifest[r['clip_id']]['target_word']]
    write(ROOT/'answers/ratings.csv', ratings, ['clip_id','criterion','rater_id','rating'])
    write(ROOT/'answers/heard_words.csv', words, ['clip_id','rater_id','heard_word'])
    write(OUT/'cleaning_log.csv', changes, ['file','line','action'])
    write(OUT/'excluded_reviews.csv', rejected, ['source_file','line','clip_id','rating_source','reason','raw_row_json'])
    write(OUT/'coverage.csv', coverage, ['file','reviews','distinct_clips','missing_clips','contributors'])
    missing = [dict(clip_id=cid,language=m['language'],available_reviews=len(reviews[cid]),needed_reviews=max(0,2-len(reviews[cid]))) for cid,m in manifest.items() if len(reviews[cid])<2]
    write(OUT/'missing_reviews.csv',missing,['clip_id','language','available_reviews','needed_reviews'])

    metrics, eligibility, audit, fixed, diagnostic, pairs = [], [], [], [], [], []
    def result(lang, measure, value, n):
        if value is not None:
            metrics.append(dict(track=1,part=4,language=lang,system='current',measure=measure,value=value,n_items=n))

    for lang in ['hi','ta']:
        ids = sorted(cid for cid,m in manifest.items() if m['language']==lang)
        paired = [cid for cid in ids if len(reviews[cid])==2]
        # Pick a fixed listener pair by coverage, before inspecting their scores.
        counts = Counter(tuple(sorted(reviews[cid])) for cid in paired)
        selected = sorted(counts, key=lambda p: (-counts[p],p))[0] if counts else ()
        for c in CRITERIA:
            usable=[]
            for cid in paired:
                rr=sorted(reviews[cid].values(),key=lambda r:int(r['coverage_set']))
                a,b=rr[0][c],rr[1][c]
                keep=abs(a-b)<=1
                audit.append(dict(clip_id=cid,language=lang,criterion=c,rater_1=rr[0]['rater_id'],rater_2=rr[1]['rater_id'],rating_1=a,rating_2=b,human_mean=(a+b)/2,prediction=scores[cid,c]['score'],eligible=keep))
                if keep: usable.append((cid,a,b,(a+b)/2,scores[cid,c]['score']))
            n=len(usable)
            qualified=n>=30 and all(len({r[j] for r in usable})>1 for j in [1,2,3])
            value=rho([r[4] for r in usable],[r[3] for r in usable]) if qualified else None
            eligibility.append(dict(language=lang,measure='agreement_'+c,available_two_ratings=len(paired),usable=n,required=30,qualified=qualified,reason='Mixed real raters across clips; development correlation, not a fixed-listener official ceiling'))
            result(lang,'agreement_'+c,value,n)
            if n>=2 and len({r[3] for r in usable})>1 and len({r[4] for r in usable})>1:
                diagnostic.append(dict(language=lang,criterion=c,pearson=float(pearsonr([r[4] for r in usable],[r[3] for r in usable]).statistic),spearman=rho([r[4] for r in usable],[r[3] for r in usable]),n_items=n))
            subset=[cid for cid in paired if tuple(sorted(reviews[cid]))==selected and abs(reviews[cid][selected[0]][c]-reviews[cid][selected[1]][c])<=1]
            a=[reviews[cid][selected[0]][c] for cid in subset]
            b=[reviews[cid][selected[1]][c] for cid in subset]
            y=[(x+z)/2 for x,z in zip(a,b)]
            valid=len(subset)>=30 and all(len(set(v))>1 for v in [a,b,y])
            fixed.append(dict(language=lang,criterion=c,rater_1=selected[0],rater_2=selected[1],pair_overlap=counts[selected],eligible_clips=len(subset),qualified=valid,spearman=rho([scores[cid,c]['score'] for cid in subset],y) if valid else None,human_ceiling=rho(a,b) if valid else None))

        word_audit=[]
        def category(value,m):
            return 'target' if value==nfc(m['target_word']) else 'contrast' if value==nfc(m['contrast']) else 'unclear' if value=='unclear' else 'invalid'
        for cid in paired:
            m=manifest[cid]
            if not m['target_word']:continue
            rr=list(reviews[cid].values())
            a,b=rr[0]['heard_word'],rr[1]['heard_word']
            p=nfc(scores[cid,'pronunciation']['predicted_word'])
            word_audit.append(dict(clip_id=cid,heard_word_1=a,heard_word_2=b,predicted_word=p,agreed=a==b,truth_category=category(a,m) if a==b else '',prediction_category=category(p,m),correct=int(p==a) if a==b else ''))
        agreed=[r for r in word_audit if r['agreed']]
        result(lang,'word_identification',sum(r['correct'] for r in agreed)/len(agreed) if len(agreed)>=10 else None,len(agreed))
        eligibility.append(dict(language=lang,measure='word_identification',available_two_ratings=len(word_audit),usable=len(agreed),required=10,qualified=len(agreed)>=10,reason='Only identical actual heard-word labels, including unclear'))
        write(OUT/f'word_items_{lang}.csv',word_audit,['clip_id','heard_word_1','heard_word_2','predicted_word','agreed','truth_category','prediction_category','correct'])
        cm=Counter((r['truth_category'],r['prediction_category']) for r in agreed)
        write(OUT/f'confusion_{lang}.csv',[dict(actual=a,**{b:cm[a,b] for b in ['target','contrast','unclear','invalid']}) for a in ['target','contrast','unclear']],['actual','target','contrast','unclear','invalid'])

        for c in ['naturalness','speaker_similarity']:
            candidate_count=0
            matching_raters=0
            shared_reference=0
            accepted=[]
            for a,b in combinations(paired,2):
                ma,mb=metadata[a],metadata[b]
                if ma['text_id']!=mb['text_id'] or ma['system_id']==mb['system_id']:continue
                if any(m['system_id'].startswith('unverified_') for m in [ma,mb]):continue
                candidate_count+=1
                if c=='speaker_similarity':
                    ra=ROOT/'inputs'/manifest[a]['reference_audio_path']; rb=ROOT/'inputs'/manifest[b]['reference_audio_path']
                    if hashlib.sha256(ra.read_bytes()).digest()!=hashlib.sha256(rb.read_bytes()).digest():continue
                shared_reference+=1
                if set(reviews[a])!=set(reviews[b]):continue
                matching_raters+=1
                delta=[reviews[a][rat][c]-reviews[b][rat][c] for rat in sorted(reviews[a])]
                if not (all(d>0 for d in delta) or all(d<0 for d in delta)):continue
                direction=1 if delta[0]>0 else -1
                gap=scores[a,c]['score']-scores[b,c]['score']
                row=dict(language=lang,criterion=c,clip_a=a,clip_b=b,rater_1=sorted(reviews[a])[0],rater_2=sorted(reviews[a])[1],human_direction=direction,predicted_difference=gap,correct=int(gap*direction>0))
                accepted.append(row);pairs.append(row)
            result(lang,'pairwise_ranking_'+c,sum(r['correct'] for r in accepted)/len(accepted) if len(accepted)>=10 else None,len(accepted))
            eligibility.append(dict(language=lang,measure='pairwise_ranking_'+c,available_two_ratings=candidate_count,usable=len(accepted),required=10,qualified=len(accepted)>=10,reason=f'{shared_reference} reference-eligible; {matching_raters} same-real-rater pairs; then strict agreement'))

    write(OUT/'correlation_items.csv',audit,['clip_id','language','criterion','rater_1','rater_2','rating_1','rating_2','human_mean','prediction','eligible'])
    write(OUT/'fixed_listener_pair_analysis.csv',fixed,['language','criterion','rater_1','rater_2','pair_overlap','eligible_clips','qualified','spearman','human_ceiling'])
    write(OUT/'correlation_diagnostics.csv',diagnostic,['language','criterion','pearson','spearman','n_items'])
    write(OUT/'ranking_pairs.csv',pairs,['language','criterion','clip_a','clip_b','rater_1','rater_2','human_direction','predicted_difference','correct'])
    write(OUT/'eligibility.csv',eligibility,['language','measure','available_two_ratings','usable','required','qualified','reason'])
    write(OUT/'results_breakdown.csv',metrics,['track','part','language','system','measure','value','n_items'])
    headline=[]
    for lang in ['hi','ta']:
        for name in ['agreement','word_identification','pairwise_ranking']:
            selected=[r for r in metrics if r['language']==lang and (r['measure']==name or r['measure'].startswith(name+'_'))]
            if selected:
                headline.append(dict(track=1,part=4,language=lang,system='current',measure=name,value=float(np.mean([r['value'] for r in selected])),n_items=sum(r['n_items'] for r in selected)))
    write(ROOT/'results.csv',headline,['track','part','language','system','measure','value','n_items'])
    write(OUT/'results.csv',headline,['track','part','language','system','measure','value','n_items'])
    summary=dict(status='development_results_only',excluded_non_independent_rows=len(rejected),reviews=len(clean),ratings=len(ratings),heard_words=len(words),missing_second_reviews=len(missing),cleaning_changes=len(changes),coverage=coverage,results=headline,breakdown=metrics,eligibility=eligibility,fixed_listener_pair_analysis=fixed,
        baseline='No separate baseline outputs or approval supplied. No baseline values, improvement, difference CI, or competition points invented.',
        listener_design='Coverage sets combine real listeners. Main correlations use per-clip means and the <=1 filter; they are descriptive development results. Fixed-pair analysis separately reports genuine listener ceilings. Coverage sets are not independent contributed datasets.',
        n_items='Agreement headline n_items is the number of eligible clip-criterion observations; a clip can count once in each of three criteria. Breakdown gives actual clips per criterion.',
        missing_data='No extrapolation or imputation. Missing second reviews do not enter two-listener metrics.',
        uncertainty='No official paired bootstrap improvement CI is computable without a separately specified baseline and official seed. Point estimates are not official competition scores.')
    (OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
