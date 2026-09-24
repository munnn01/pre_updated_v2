# V2 paper-validation plan (exploratory reanalysis)

This plan specifies the analysis implemented in `ops/paper_validation.py`.
It reuses the already collected 1,000 paired TEST clips per codec; it is
**not** a new holdout and must not be described as independent confirmation.

## Question and unit

Does frozen dual-analyzer mode C improve the rate--accuracy trade-off over
simple fixed transforms and the frozen A/B policies? The independent sampling
unit for uncertainty is a source video. All five QPs, six candidates, policies,
and analyzers are repeated measurements of that video, not new samples.

## Fixed comparisons

For each codec separately, use the same 1,000 video IDs and existing real-codec
candidate measurements. Compare the unmodified `identity128` anchor with all
five non-identity fixed transforms, V1, and the pilot-frozen A, B, C policies.
Report every arm, including failures; do not choose a winning fixed transform
from TEST and then call it a prespecified comparator. Also compare C directly
against B with paired whole-video bootstrap resampling.

Report Top-1 BD-rate, BD-accuracy (percentage points), each QP's bitrate and
Top-1 accuracy, the minimum same-QP accuracy gap, choice counts, and 95%
bootstrap intervals. Bootstrap whole videos with all their QPs paired. The
bootstrap is descriptive because this TEST set influenced earlier V1 research.

## Independent experiments still required

1. Evaluate a frozen third analyzer (`mc3_18`) that never enters policy fitting,
   calibration, or selection. Decode the bitstream actually chosen by the frozen
   policy; do not select using the third model or its labels.
2. Repeat on a genuinely new source-disjoint holdout/dataset. The unused tail
   of the current TEST split is only a diagnostic if it has been inspected in
   earlier work; it is not automatically a pristine holdout.
3. Record encoder, decoder, analyzer, and selection wall times and peak memory
   on one specified machine. Include every trial encoding in total cost.

These three items are prospective. This reanalysis cannot satisfy them by
recomputing statistics from the existing two-analyzer JSONL records.

## Implemented follow-up runners (not yet measured)

`ops/paper_heldout_mc3.py` runs the frozen C selector from the cached
two-analyzer measurements, then **re-encodes only the identity and chosen
streams** and evaluates `mc3_18`. It never lets mc3 predictions or labels
affect selection. Run two 500-clip shards per codec and merge their records;
the shard BD-rates are diagnostic only. Both prior V2 cache directories and
the Kinetics index/videos must be mounted in the runtime environment.

```powershell
python -m ops.paper_heldout_mc3 evaluate --index <kinetics-index.json> `
  --codec h264 --shard 0 `
  --cache-dir <h264-original-shard-0> --cache-dir <h264-original-shard-1> `
  --out-dir <mc3-h264-shard-0-output>
python -m ops.paper_heldout_mc3 merge --codec h264 `
  --shard-dir <mc3-h264-shard-0-output> `
  --shard-dir <mc3-h264-shard-1-output> `
  --out <mc3-h264-merged.json> --bootstrap 2000
```

Repeat with shard 1 and H.265. A mode-C gain on mc3 would support transfer
to one unseen analyzer, not universal model independence. The sample remains
the previously inspected 1,000 TEST clips.

`ops/paper_runtime.py` benchmarks the *complete* six-candidate selector
against an identity-only encoder on the same predetermined source clips.
The clip is the paired block; arm order is balanced by a fixed hash. It times
all trial encodes, the two feature analyzers and the selector, excluding model
startup/download. Report both absolute wall time and overhead ratio.

```powershell
python -m ops.paper_runtime --index <kinetics-index.json> `
  --codec h264 --split val --clips 20 --out-dir <runtime-h264-output>
```

The CLI examples describe how to run the follow-ups; their numerical outcomes
must not be claimed until the jobs finish and the results have been audited.
