# Latest input audit

The file named ta_set_2_completed.csv does not add new independent human reviews.
Its 62 original_set_2 rows exactly match the earlier second-set export. Its 58
reference_rating_from_set_1 rows have blank listener IDs/timestamps and exactly
copy Set 1's criterion ratings and heard words. They remain in the raw input and
excluded_reviews.csv, but do not count as a second person. No labels were invented.
The calculated scores therefore remain identical to the previous genuine-data run.

Run from C:/Users/user/Desktop/aadhi/part4:
`python main.py --human-evaluation`

Final Part 4 handoff materials are in outputs/midterm_submission/. They reflect
available evidence, with missing coverage and baseline limitations disclosed.
This does not make an incomplete four-part team submission complete.

---

# Read this first - Part 4 handoff

## What to send to your leader
The `outputs/midterm_submission/` folder contains the three Part 4 contributions to the midterm:
- `report.pdf`: a readable Part 4 report; merge/summarize it into the TEAM report.
- `results.csv`: current development results, in seven columns. Merge into the TEAM sheet.
- `test_set.zip`: part4/inputs and part4/answers/clip_metadata.csv. Merge its part4
  folder with the other parts into the TEAM test_set.zip.

Do NOT upload this whole project ZIP as an extra midterm deliverable. The PDF
accepts report.pdf (at most 6 pages plus references), results.csv and test_set.zip
for Track 1 midterm. code.zip/freeze.txt/outputs.zip are different later milestones.
The six-page limit is for the ENTIRE team report, not six pages per part.

This is the most complete evidence-backed Part 4 handoff possible from the supplied
files, not a claim that all team submission requirements have been satisfied.
Before sending: confirm human-recording consent and audio redistribution rights,
check the 20 human scripts with native speakers, resolve the missing approved
baseline, and merge the other parts. Shared-reference speaker ranking remains absent.

## What I did
1. Preserved the latest four supplied CSVs byte-for-byte in answers/raw. The completed Tamil export contains 62 originals and 58 copied rows; the copied rows are separately audited and excluded.
2. Checked clip IDs, languages, coverage-set numbers, distinct raters, integer
   ratings in 1..5, candidate words and duplicates. All 422 independent reviews were valid;
   no rating corrections were necessary. No records or values were invented.
3. Converted the wide website rows into 1,266 long-form criterion ratings and
   278 heard-word observations. Real listener IDs and disagreements are retained.
4. Matched those answers to your existing 720 model predictions. Predictions,
   transcription logic and audio-model code were not changed or rerun.
5. Calculated filtered Spearman correlations, supplementary Pearson correlations,
   genuine fixed-listener-pair correlations/ceilings, word identification accuracy,
   confusion matrices, and pairwise eligibility/diagnostics.
6. Recorded every exclusion and the 58 Tamil clips needing second reviews.
7. Kept your old strict scorer intact; added evaluate_humans.py for these partial,
   mixed-listener DEVELOPMENT exports rather than bypassing its official-data checks.
8. Kept all 240 development clips in . For the contributed test ZIP only,
   excluded 40 clips whose TTS providers are unknown. The remaining 100 clips per
   language include 30 Microsoft Edge, 30 Google Translate, 30 Meta MMS and 10 human;
   60 per language are minimal-pair items. All clips meet the 20-second maximum.
9. Added a report, file-by-file README, reproducible metric command and checklist.

## Explain the pipeline
The audio evaluator makes predictions. Human evaluation checks those predictions.
Whisper small transcribes generated audio; 1-CER compares that with intended text.
It also predicts target/contrast/unclear by looking for candidate tokens.
WavLM compares generated and reference speaker embeddings using cosine similarity.
UTMOS predicts naturalness. These produce scores.csv: 240 clips x 3 criteria.

Humans give independent 1-5 ratings and select what word they heard. For each
criterion, take clips with two ratings differing by no more than one. Average
the two ratings and correlate the model's ordering with the human ordering.
Spearman 1 means identical ordering, 0 means no monotonic agreement, and a negative
number means a tendency to reverse the human ordering. This is not percent accuracy.

For words, keep only clips where both people heard the same candidate/unclear.
Compare that actual label with the model label. Correct / eligible is accuracy.
The confusion table is the detailed error breakdown, not a fourth required metric.
It is multiclass, so there is no single universal false-positive/true-negative
table unless you explicitly choose one class versus the others.

For pairwise ranking, both of the SAME real people must prefer A over B (or B
over A), for two systems saying the same text. Model ties count wrong. Speaker
pairs additionally need the same target reference. Eight Hindi and three Tamil
naturalness pairs pass, below ten each; no speaker pairs pass. Unavailable != zero.

## Main results to explain
Hindi Spearman: naturalness 0.0552 (68 clips), pronunciation 0.1890 (84),
speaker similarity 0.1940 (89). Criterion mean 0.1460.
Tamil: naturalness -0.1980 (49), pronunciation -0.0819 (53),
speaker similarity 0.2557 (52). Criterion mean -0.0081.
Word identification: Hindi 45/73 = 61.64%; Tamil 18/34 = 52.94%.
These are early, weak results. They justify improvement work, not a success claim.
Agreement headline n_items sums eligible clip-criterion observations, not people
or unique audio files; use results_breakdown.csv for per-criterion counts.

Coverage sets combine people. A Set 1/Set 2 correlation would incorrectly pretend
each is one person. Instead, fixed_listener_pair_analysis.csv selects the two real
people with greatest overlap, based only on coverage, and computes actual human
ceilings on qualifying subsets. These subsets differ from the headline corpus.
The mixed-rater headline correlations must be labeled as development estimates.

## Why Tamil was not extrapolated
There are 62 real second reviews and 58 missing. Duplicating Set 1 or using model
predictions would manufacture independent human evidence and artificially affect
agreement/accuracy. The PDF says the organizers label contributed midterm audio;
your partial real development results can be disclosed. Missing reviews remain missing.

## Baseline and confidence intervals
No separate baseline outputs or judge approval were supplied. The existing public-
model combination is a running baseline-style system, but that is not proof of an
approved scored comparator. results.csv has current rows only. Ask the leader to
resolve baseline approval/reporting; do not duplicate current rows under another
name and imply a separate experiment. There is no valid x-minus-baseline confidence
interval or competition point calculation to report from the supplied evidence.
The previous strict score_results.py contains the prescribed paired-bootstrap
machinery for a future complete, correctly specified comparison.

## Requirements checklist
- [x] Checked Rime pp. 18-21, 30-31, 37-40 and bootstrap appendix.
- [x] Command-line inference source and required prediction columns preserved.
- [x] 720 original scores; no missing clip/criterion outputs.
- [x] Input audio and reference audio kept inside inputs/.
- [x] Source metadata separated into answers/clip_metadata.csv.
- [x] Contributed set: 100 Hindi and 100 Tamil clips, each <=20 seconds.
- [x] Three named TTS systems plus 10 human clips per language.
- [x] At least 20 minimal-pair items: 60 per language in contributed subset.
- [x] Same-text outputs across three known TTS systems and consistent text IDs.
- [x] Actual words, real rater IDs and disagreements retained.
- [x] Spearman, word accuracy, error categories and coverage reported.
- [x] Proposed controlled comparison and revised plan in report.
- [x] Required seven-column results layout; unknown metrics not filled with zero.
- [x] No institution name inserted in submission/report.pdf.
- [ ] Tamil's 58 missing second reviews still needed for full development coverage.
- [ ] Human scripts, audio consent and redistribution permissions need team confirmation.
- [ ] No qualifying cross-system common-reference speaker-ranking data.
- [ ] Naturalness ranking needs at least 10 usable pairs per language.
- [ ] Separate approved baseline/comparison outputs and improvement CIs absent.
- [ ] The team must merge Parts 1-3 and fit the combined six-page report limit.
- [ ] Fresh-machine model inference and final frozen-code release not validated here.

## Working directory
Your original C:/Users/user/Desktop/aadhi/part4 project is now the active project. main.py has a --human-evaluation mode and evaluate_humans.py is added. The model-scoring modules and saved predictions remain unchanged. The previous main.py and README are backed up in outputs/_archive/before_human_evaluation_integration/. The full ZIP is for your leader/backup; only the merged
three midterm deliverables should go to organizers.
