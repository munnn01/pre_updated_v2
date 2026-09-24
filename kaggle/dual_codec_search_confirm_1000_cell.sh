set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
CODEC="__CODEC__"
SHARD="__SHARD__"
REPO="/kaggle/working/pre_updated_v2"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/dual_codec_search_v2_confirm_1000/${CODEC}/shard_${SHARD}"
mkdir -p "$ROOT_OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "dual_v2_confirm_${CODEC}_shard_${SHARD}.tgz" \
    "outputs/dual_codec_search_v2_confirm_1000/${CODEC}/shard_${SHARD}" \
    kinetics_hash_split.json 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/dual_v2_confirm_${CODEC}_shard_${SHARD}.tgz"
  exit "$rc"
}
trap finish EXIT
if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/munnn01/pre_updated_v2.git "$REPO"
fi
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision, cv2; print("torch", torch.__version__, "torchvision", torchvision.__version__, "opencv", cv2.__version__, "cuda", torch.cuda.is_available()); assert torch.cuda.is_available(), "GPU is required"'
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265'
KIN_ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then
    KIN_ROOT="$candidate"
    break
  fi
done
if [ -z "$KIN_ROOT" ]; then
  echo "ERROR: attach qktttttttttt/kineticscleaned" >&2
  exit 2
fi
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" \
  --assert-fingerprint 30f083f8520a
cp docs/RUN_DESIGN_DUAL_CODEC_SEARCH_V2_CONFIRM_1000.md "$ROOT_OUT/preregistered_design.md"
cp configs/dual_codec_search_v2_confirm_1000.json "$ROOT_OUT/frozen_config.json"
python ops/dual_codec_search_confirm_1000.py \
  --index "$INDEX" --codec "$CODEC" --shard "$SHARD" --out-dir "$ROOT_OUT" \
  2>&1 | tee "$ROOT_OUT/run.log"
test -f "$ROOT_OUT/shard_result.json"
test -f "$ROOT_OUT/shard_records.jsonl"
echo "[done] frozen V2 confirm codec=$CODEC shard=$SHARD"
