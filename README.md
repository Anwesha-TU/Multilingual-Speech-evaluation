# Part 4 speech evaluation

## Generate the human-evaluation results
From this folder, run:

```
python main.py --human-evaluation
```

This uses your existing 720 model predictions, so it does not rerun Whisper,
WavLM or UTMOS. It reads the four named CSV exports in answers/raw/ and writes
results.csv plus outputs/evaluation/ (metrics, coverage, confusion matrices,
per-item audit, listener ceilings and exclusions). Dependencies numpy and scipy
are already included in requirements.txt. You can also run evaluate_humans.py.

## Generate audio-model predictions (existing behavior)
```
python main.py --require-complete
```
This reads inputs/manifest.csv and writes outputs/current/scores.csv.
scores.csv = per-clip predictions. results.csv = measured agreement with people.
Model code and the previously generated predictions have not been altered.

## Supplied completed export
ta_set_2_completed.csv contains 62 original second reviews and 58 rows explicitly
marked reference_rating_from_set_1, without listener IDs. These 58 copied rows
are preserved in the raw file and outputs/evaluation/excluded_reviews.csv, but
are NOT treated as independent human judgments. The valid collection remains
422 reviews and Tamil still needs 58 second reviews. No values are fabricated.

Replace answers/raw/ta_set_2_completed.csv with an actual completed export to
rerun. Genuine reviews need real listener IDs and original provenance (or no
rating_source column). Do not relabel copied rows as genuine observations.

## Submission handoff
outputs/midterm_submission/ contains the Part 4 report.pdf, results.csv and
test_set.zip. The report is four content pages plus references. Your leader must
merge Parts 1-3 and respect the six-page TEAM report limit. The contributed ZIP
has 100 known-source clips per language: 30 Edge, 30 gTTS, 30 MMS, 10 human.
All 120 development clips per language remain in inputs/ for reproducibility.
The 40 extra development clips with unknown source providers were not contributed.

Read outputs/evaluation/HANDOFF.md for methods, results and outstanding checks.
The full project handoff ZIP is for your leader, not an extra midterm upload.
The report and package must be refreshed if new genuine reviews change results.

## Limitations
No distinct approved baseline predictions were supplied; no improvement, paired
difference CI or competition points are claimed. Mixed-listener correlations are
descriptive development results; actual fixed-pair ceilings are reported separately.
Naturalness ranking has fewer than ten usable pairs in each language. Speaker
ranking has no qualifying cross-system common-reference pairs. These measures
are unavailable, not zero. Human scripts and recording permissions need team review.

The original strict score_results.py remains available for complete official
comparison data. Observed model versions, hashes and patched SpeechMOS code are
under outputs/reproducibility/. New-machine inference still requires provisioning
model weights and updating recorded local cache locations. The human-scoring
command only needs the included predictions and numpy/scipy.

Pre-integration main.py and README.md are recoverable under
outputs/_archive/before_human_evaluation_integration/.
