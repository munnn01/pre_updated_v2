# Frozen V2 dual-analyzer, 1,000-video paired TEST comparison

Protocol fixed before collecting the V2 1,000-video results. The V2 pilot selected one policy per codec on TRAIN fit/calibration, then measured 200 VAL-development clips. This run does **not** fit the risk model, adjust thresholds, or inspect TEST outcomes before choosing a policy.

## Question and limitation

Can one frozen selected bitstream improve both `r2plus1d_18` and `r3d_18` at comparable rate on either H.264 or H.265? Both analyzers influenced V2 selection and are targets, not held-out backbones. The 1,000-video TEST sample is exactly the sample used for the prior V1 release (`aae3888f3ae34d08`), permitting paired V1 comparison. That TEST result was inspected before this experiment, so this is a **replication/comparison on a previously inspected set**, not a fresh independent confirmation. No retuning after these results may be reported as independent success.

## Frozen inputs and design

- Code pinned to the new full Git SHA used by the notebook. Pilot risk coefficients and policies are extracted byte-for-byte from the two completed pilot output archives; expected raw-file SHA-256 values are in `configs/dual_codec_search_v2_confirm_1000.json`.
- 1,000 distinct TEST videos chosen by the existing V1 frozen salt, with exactly the same ordered IDs and fingerprint for H.264 and H.265. TRAIN/VAL/TEST membership derives from canonical `kineticscleaned` hash split (test fingerprint prefix `30f083f8520a`).
- Six existing candidates; QP 30/35/40/45/50; 16 RGB frames, stride 2, 128-pixel source, `medium` FFmpeg H.264/H.265, normalized bpp against original 128×128 pixels. Pretrained frozen torchvision KINETICS400_V1 `r2plus1d_18` and `r3d_18` evaluate the *same decoded stream*.
- Arms: identity control, V1 policy, selected V2 policy C. Every video/QP receives all six candidates. The selector gets a strict allowlist of label-free signals; labels are read only after selection for accuracy.
- Source video is the unit of replication. Report aggregate five-QP rate/Top-1 curves, BD-rate and BD-accuracy for each analyzer, minimum same-QP Top-1 gap, candidate counts and paired 2,000-draw whole-video bootstrap 95% intervals for both metrics.
- Main criterion: for **at least one codec**, both analyzers have Top-1 BD-rate strictly below −15% and BD-accuracy above zero. Secondary guard: minimum same-QP Top-1 gap at least −1 percentage point for each analyzer. Report the point criterion, confidence intervals and guard separately.

The 1,000-video sample is split into two disjoint shards of 500 for each codec to reduce Kaggle wall time. Shards only partition computation: merge every per-video record first, then calculate one aggregate curve and whole-video bootstrap. Do not average shard BD-rates.

## Outputs and fail-closed checks

Each notebook writes `manifest.json`, `frozen_policy.json`, `risk_model.json`, `cache/test/*.json`, `shard_records.jsonl`, `shard_result.json` and an archive. The merger rejects duplicate/missing videos, mismatched code/config/frozen hashes, QP/candidate order, missing analyzer outcomes or inconsistent policies. Final output keeps all point metrics and bootstrap intervals. No source videos, weights, API tokens or raw private credentials are packaged.
