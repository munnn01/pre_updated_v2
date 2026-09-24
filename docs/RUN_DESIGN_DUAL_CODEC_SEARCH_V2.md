# Dual-analyzer codec search V2 — development pilot

Protocol fixed before V2 pilot collection. NOT a new 1,000-video confirmation;
there is no guaranteed BD-rate outcome.

## Objective and controls

One selected bitstream per source video/QP serves both frozen torchvision
KINETICS400_V1 analyzers r2plus1d_18 and r3d_18. Both are target models, neither
is held-out. Final target: negative Top-1 BD-rate and positive BD-accuracy for
both, with at least one codec below -15% on BOTH models.

Concurrent control V1: six candidates identity128, area112, area96,
area112_up128, blur020_128, blur040_128; QP 30/35/40/45/50; 16 RGB frames,
stride 2, center temporal crop; medium libx264/libx265, yuv420p; original
128x128 bpp denominator. No temporal changes or analyzer fine-tuning. All
arms reuse the SAME real encoded/decoded candidates and both models see
identical decoded pixels. V1 files/results remain unchanged in this fork.

## Arms and fixed search grid

- V1: primary KL slack .1, feature slack .05, source confidence guard .6.
- A: V1 choice followed by r3d safeguard; rejection returns identity.
- B: lowest bitrate of all six candidates passing both model safeguards.
  Each model compares against its OWN source/anchor, not the other model.
- C: logistic harm predictor per analyzer trained on the event `anchor
  correct AND candidate wrong`. Explicit label-free features: QP, rate ratio,
  candidate identity, source/decoded confidence, margin, entropy, source/anchor
  agreement and semantic distances. Primary V1 guard remains active; BOTH
  predicted risks must be below the threshold for the current QP group.

C uses StandardScaler + LogisticRegression(C=1, max_iter=2000), fitted on
TRAIN-fit only, no class reweighting. Numeric JSON export, no executable pickle.
Scores are NOT claimed to be calibrated probabilities. Single-class outcomes
trigger conservative fallback. A/B cross confidence thresholds {.6,.8,1.01},
where 1.01 disables the source argmax guard; (KL,feature) slacks (0,0),
(.02,.005),(.05,.02),(.1,.05). C thresholds {.02,.05,.10,.20} independently
for QP<=35 and QP>=40. Total 12 A, 12 B, 16 C configurations.

## Splits and leakage controls

Canonical kineticscleaned hash split (TEST fingerprint 30f083f8520a), but NO
TEST videos are decoded. Each codec: 400 TRAIN-fit, 200 disjoint
TRAIN-calibration, 200 VAL-development; same deterministic IDs for both codecs.
Different salted class-balanced selections without replacement; known Kinetics
11-character source IDs group temporal excerpts. Otherwise use class/filename.
Exclude TRAIN sources appearing in VAL/TEST. This cannot detect unknown
re-encoded duplicate sources: final confirmation needs a further source audit.

Save exact split IDs, fingerprints, code SHA and versions. Strip outcomes,
labels, target-class probabilities and IDs before inference. Fit scaler and
classifiers only on 400 TRAIN-fit. Calibrate only on 200 TRAIN-calibration.
Minimize WORST analyzer BD-rate subject to both BD-rates<0, both
BD-accuracies>0 and minimum same-QP Top-1 difference >= -1 percentage point
for each analyzer. If no configuration is eligible, freeze identity explicitly.
Choose one policy per arm AND a global winner on calibration BEFORE seeing DEV.
Report all DEV arms, but do not change the winner after looking at DEV.

VAL and the previous 1,000 TEST clips have already been inspected in the project.
These are exploratory DEV results, NOT independent confirmation. A promising
pilot must be frozen and validated on audited unused sources before a final claim.
There is no ground-truth oracle in deployed selection.

## Analysis, runtime and artifacts

Compute curves from whole clips, never average shard BD-rates. Keep V1's
polynomial BD method on overlapping integration ranges and report all five
points. BD-rate in %, BD-accuracy in fraction AND percentage points. Paired
bootstrap: 500 draws of whole videos, preserving all QPs; report both metric
intervals and valid draw counts. Pilot intervals are descriptive, not adjusted
for multiple comparisons or a confirmation claim.

Two GPU notebooks, one per codec; A/B/C share all encodes. No custom wall-clock
timeout; Kaggle platform limits still apply. Atomic per-clip caches and strict
manifest checks permit within-session resume. A new session must attach/copy
prior outputs to reuse them. Save model/calibration/frozen policy BEFORE DEV.
EXIT trap packs partial output on ordinary shell failures; hard platform kills
may still lose artifacts. Never package credentials, source videos or weights.

Outputs: manifest.json, cache/{fit,calibration,dev}/*.json, risk_model.json,
calibration.json, frozen_policy.json, pilot_result.json, run.log, and archive
dual_codec_search_v2_<codec>.tgz.

```bash
python ops/dual_codec_search.py --index kinetics_hash_split.json --codec h264 \
  --out-dir outputs/dual_codec_search_v2/h264
python ops/push_dual_codec_search.py --commit FULL_GIT_SHA --account ACCOUNT \
  --codec h264 --pool /private/path/pool.json
```

Template: kaggle/dual_codec_search_cell.sh. Clone pre_updated_v2 at exact SHA;
attach qktttttttttt/kineticscleaned, enable Internet + GPU. Never embed pool.json
or tokens in notebooks. V1 history/results remain historical artifacts.
