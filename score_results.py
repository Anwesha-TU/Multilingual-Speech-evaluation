"""Part 4 scoring: strict output validation, eligibility, diagnostics, paired CI.

Config is a JSON list of {team_id, package, scores, baseline_scores} records.
Paths resolve against the config file. This script never loads audio models.
Human answer files must contain real independent ratings, not blank review forms.
"""
import argparse,csv,hashlib,itertools,json,math,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

CRITERIA=('naturalness','pronunciation','speaker_similarity')

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def unique(rows,fields):
    out={}
    for r in rows:
        key=tuple(r[k] for k in fields)
        if key in out:raise ValueError(f'Duplicate {fields}: {key}')
        out[key]=r
    return out

def word(s):return unicodedata.normalize('NFC',s.strip())

def correlation(x,y):
    if len(set(x))<2 or len(set(y))<2:return 0.0
    return float(spearmanr(x,y).statistic)

def load_scores(path,manifest):
    rows=read(path);scores=unique(rows,['clip_id','criterion'])
    expected={(cid,c) for cid in manifest for c in CRITERIA}
    if set(scores)!=expected:
        raise ValueError(f'{path}: missing {len(expected-set(scores))}, unexpected {len(set(scores)-expected)} score rows; incomplete outputs cannot be scored')
    for key,r in scores.items():
        r['score']=float(r['score'])
        if not math.isfinite(r['score']):raise ValueError('Nonfinite score')
    for cid,m in manifest.items():
        if m['target_word']:
            predictions=[word(scores[cid,c].get('predicted_word','')) for c in CRITERIA]
            if not predictions[1]:raise ValueError(f'Missing pronunciation predicted_word: {cid}')
            if any(v and v!=predictions[1] for v in predictions):raise ValueError('Inconsistent word predictions across criteria')
    return scores

def prepare(config,base):
    groups=[];word_items=[];pair_items=[];coverage=[];diagnostics=[];teams=set()
    for ds in config:
        team=ds['team_id']
        if team in teams or '/' in team:raise ValueError('Team IDs must be unique and contain no slash')
        teams.add(team);root=(base/ds['package']).resolve()
        manifest={k[0]:v for k,v in unique(read(root/'inputs/manifest.csv'),['clip_id']).items()}
        metadata={k[0]:v for k,v in unique(read(root/'answers/clip_metadata.csv'),['clip_id']).items()}
        if set(metadata)!=set(manifest):raise ValueError('Metadata and manifest clip IDs differ')
        text_groups=defaultdict(set)
        from text_methods import normalize_text as normalize
        for cid,m in manifest.items():
            language={'Hindi':'hi','Tamil':'ta'}.get(m['language'],m['language'])
            if language not in ('hi','ta'):raise ValueError('Invalid manifest language')
            md=metadata[cid]
            if not md['text_id'] or not md['system_id'] or md['system_id'].startswith('unverified_'):
                raise ValueError('Confirm source systems and text IDs before scoring official metrics')
            text_groups[md['text_id']].add((language,normalize(m['text'])))
            if bool(m['target_word'])!=bool(m['contrast']) or (m['target_word'] and word(m['target_word'])==word(m['contrast'])):
                raise ValueError('Minimal-pair candidates must be two distinct actual words')
        if any(len(v)>1 for v in text_groups.values()):raise ValueError('Different texts share a text_id')
        pred=load_scores(base/ds['scores'],manifest);baseline=load_scores(base/ds['baseline_scores'],manifest)
        ratings=defaultdict(dict)
        for r in read(root/'answers/ratings.csv'):
            key=(r['clip_id'],r['criterion']);rat=r['rater_id']
            if key not in pred or not rat or rat in ratings[key]:raise ValueError('Invalid/duplicate rating')
            val=float(r['rating'])
            if not math.isfinite(val) or not 1<=val<=5:raise ValueError('Rating must be in 1..5')
            ratings[key][rat]=val
        if set(ratings)!=set(pred) or any(len(v)!=2 for v in ratings.values()):raise ValueError('Each clip/criterion needs exactly two independent ratings')
        heard=defaultdict(dict)
        for r in read(root/'answers/heard_words.csv'):
            cid=r['clip_id'];rat=r['rater_id']
            if cid not in manifest or not manifest[cid]['target_word'] or not rat or rat in heard[cid]:raise ValueError('Invalid/duplicate heard-word label')
            m=manifest[cid];label=word(r['heard_word'])
            if label not in {word(m['target_word']),word(m['contrast']),'unclear'}:raise ValueError(f'Invalid human heard-word label: {cid}')
            heard[cid][rat]=label
        candidates={cid for cid,m in manifest.items() if m['target_word']}
        if set(heard)!=candidates or any(len(v)!=2 for v in heard.values()):raise ValueError('Each minimal-pair item needs two heard-word labels')
        for language in ('hi','ta'):
            ids=[cid for cid,m in manifest.items() if {'Hindi':'hi','Tamil':'ta'}.get(m['language'],m['language'])==language]
            for criterion in CRITERIA:
                eligible=[]
                rater_ids={tuple(sorted(ratings[cid,criterion])) for cid in ids}
                if len(rater_ids)>1:raise ValueError('This implementation requires the same two raters within each set/language/criterion for listener ceiling')
                for cid in ids:
                    rr=[v for _,v in sorted(ratings[cid,criterion].items())]
                    if abs(rr[0]-rr[1])<=1:
                        eligible.append({'id':f'{team}/{cid}','x':pred[cid,criterion]['score'],'b':baseline[cid,criterion]['score'],'y':sum(rr)/2,'r1':rr[0],'r2':rr[1]})
                qualifies=len(eligible)>=30 and all(len({r[k] for r in eligible})>1 for k in ('r1','r2','y'))
                coverage.append(dict(team_id=team,language=language,measure='agreement',criterion=criterion,eligible=len(eligible),qualified=qualifies))
                if qualifies:groups.append(dict(team=team,language=language,criterion=criterion,items=sorted(eligible,key=lambda r:r['id'].encode())))
            wi=[]
            for cid in ids:
                if cid not in heard:continue
                labels=list(heard[cid].values())
                if labels[0]!=labels[1]:continue
                expected=labels[0];m=manifest[cid]
                def category(v):return 'target' if v==word(m['target_word']) else 'contrast' if v==word(m['contrast']) else 'unclear' if v=='unclear' else 'invalid'
                x=word(pred[cid,'pronunciation']['predicted_word']);b=word(baseline[cid,'pronunciation']['predicted_word'])
                wi.append(dict(id=f'{team}/{cid}',language=language,x=int(x==expected),b=int(b==expected),truth=category(expected),prediction=category(x),expected_word=expected,predicted_word=x))
            qualifies=len(wi)>=10
            coverage.append(dict(team_id=team,language=language,measure='word_identification',eligible=len(wi),qualified=qualifies))
            diagnostics.extend(wi)
            if qualifies:word_items.extend(wi)
            for criterion in ('naturalness','speaker_similarity'):
                pi=[]
                bytext=defaultdict(list)
                for cid in ids:bytext[metadata[cid]['text_id']].append(cid)
                for same_text in bytext.values():
                    for a,b in itertools.combinations(sorted(same_text,key=lambda x:x.encode()),2):
                        if metadata[a]['system_id']==metadata[b]['system_id']:continue
                        if criterion=='speaker_similarity':
                            ra=manifest[a]['reference_audio_path'];rb=manifest[b]['reference_audio_path']
                            if not ra or not rb:continue
                            pa=(root/'inputs'/ra).resolve();pb=(root/'inputs'/rb).resolve()
                            if not pa.is_file() or not pb.is_file():raise FileNotFoundError('Missing reference file')
                            if pa!=pb and hashlib.sha256(pa.read_bytes()).digest()!=hashlib.sha256(pb.read_bytes()).digest():continue
                        ar=ratings[a,criterion];br=ratings[b,criterion]
                        if set(ar)!=set(br):continue
                        delta=[ar[r]-br[r] for r in ar]
                        if not (all(d>0 for d in delta) or all(d<0 for d in delta)):continue
                        direction=1 if delta[0]>0 else -1
                        pi.append(dict(id='~'.join(sorted([f'{team}/{a}',f'{team}/{b}'],key=lambda x:x.encode())),language=language,criterion=criterion,x=int((pred[a,criterion]['score']-pred[b,criterion]['score'])*direction>0),b=int((baseline[a,criterion]['score']-baseline[b,criterion]['score'])*direction>0)))
                qualifies=len(pi)>=10
                coverage.append(dict(team_id=team,language=language,measure='pairwise_ranking',criterion=criterion,eligible=len(pi),qualified=qualifies))
                if qualifies:pair_items.extend(pi)
    return groups,sorted(word_items,key=lambda r:r['id'].encode()),sorted(pair_items,key=lambda r:r['id'].encode()),coverage,diagnostics

def draws(items,seed,key,replicate):
    n=len(items)
    return [items[int.from_bytes(hashlib.sha256(f'{seed}|bootstrap|{key}|{replicate}|{j}'.encode()).digest(),'big')%n] for j in range(n)]

def agreement(groups,column,seed=None,replicate=None):
    cells=defaultdict(list)
    for g in groups:
        values=g['items']
        if replicate is not None:values=draws(values,seed,f"part4_agreement/{g['team']}/{g['language']}/{g['criterion']}",replicate)
        score=correlation([r[column] for r in values],[r['r2'] if column=='r1' else r['y'] for r in values])
        cells[g['language'],g['criterion']].append(score)
    return float(np.mean([np.mean(v) for v in cells.values()])) if cells else None

def accuracy(items,column):return float(np.mean([r[column] for r in items])) if items else None

def ranking(items,column,seed=None,replicate=None):
    values=[]
    for c in ('naturalness','speaker_similarity'):
        selected=[r for r in items if r['criterion']==c]
        if not selected:continue
        if replicate is not None:selected=draws(selected,seed,f'part4_pairwise_ranking/{c}',replicate)
        values.append(accuracy(selected,column))
    return float(np.mean(values)) if values else None

def compute(config,base,seed):
    groups,words,pairs,coverage,diagnostics=prepare(config,base)
    result={'coverage':coverage,'metrics':{},'human_correlation_ceiling':agreement(groups,'r1'),'word_errors':[r for r in diagnostics if not r['x']]}
    for name in ('agreement','word_identification','pairwise_ranking'):
        def value(col,r=None):
            if name=='agreement':return agreement(groups,col,seed,r)
            if name=='pairwise_ranking':return ranking(pairs,col,seed,r)
            data=draws(words,seed,'part4_word_identification',r) if r is not None and words else words
            return accuracy(data,col)
        x=value('x');b=value('b')
        if x is None:result['metrics'][name]={'status':'no_qualifying_data'};continue
        differences=sorted(value('x',r)-value('b',r) for r in range(1,1001))
        result['metrics'][name]={'current':x,'baseline':b,'difference':x-b,'difference_ci95':[differences[24],differences[974]]}
    matrices={}
    for lang in ('hi','ta'):
        counts=Counter((r['truth'],r['prediction']) for r in diagnostics if r['language']==lang)
        matrices[lang]={a:{b:counts[a,b] for b in ('target','contrast','unclear','invalid')} for a in ('target','contrast','unclear')}
    result['confusion_matrices']=matrices
    # Export language-level result rows; pooled bootstrap metrics remain in JSON.
    export=[]
    for lang in ('hi','ta'):
        for system,col in [('baseline','b'),('current','x')]:
            for c in CRITERIA:
                gs=[g for g in groups if g['language']==lang and g['criterion']==c]
                if gs:export.append(dict(track=1,part=4,language=lang,system=system,measure=f'agreement_{c}',value=agreement(gs,col),n_items=sum(len(g['items']) for g in gs)))
            ws=[r for r in words if r['language']==lang]
            if ws:export.append(dict(track=1,part=4,language=lang,system=system,measure='word_identification',value=accuracy(ws,col),n_items=len(ws)))
            for c in ('naturalness','speaker_similarity'):
                ps=[r for r in pairs if r['language']==lang and r['criterion']==c]
                if ps:export.append(dict(track=1,part=4,language=lang,system=system,measure=f'pairwise_ranking_{c}',value=accuracy(ps,col),n_items=len(ps)))
    result['results_rows']=export
    result['note']='Raw measures only, not normalized competition points. Detailed measure suffixes label language/criterion breakdowns; confirm official results-sheet naming before submission.'
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--seed',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    path=Path(a.config).resolve();result=compute(json.loads(path.read_text()),path.parent,a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'results_breakdown.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['track','part','language','system','measure','value','n_items']);w.writeheader();w.writerows(result['results_rows'])
    print(json.dumps(result['metrics'],indent=2))
