set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
CODEC="__CODEC__"
REPO="/kaggle/working/pre_updated_v2"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/dual_codec_search_v2/${CODEC}"
mkdir -p "$ROOT_OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "dual_codec_search_v2_${CODEC}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/dual_codec_search_v2_${CODEC}.tgz"
  exit "$rc"
}
trap finish EXIT
if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/munnn01/pre_updated_v2.git "$REPO"
fi
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision, sklearn; print("torch", torch.__version__, "torchvision", torchvision.__version__, "sklearn", sklearn.__version__, "cuda", torch.cuda.is_available()); assert torch.cuda.is_available(), "GPU is required for this pilot"'
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
python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX" --assert-fingerprint 30f083f8520a
cp docs/RUN_DESIGN_DUAL_CODEC_SEARCH_V2.md "$ROOT_OUT/preregistered_design.md"
python ops/dual_codec_search.py \
  --index "$INDEX" --codec "$CODEC" --out-dir "$ROOT_OUT" \
  --fit-clips 400 --calibration-clips 200 --dev-clips 200 --bootstrap 500 \
  2>&1 | tee "$ROOT_OUT/run.log"
test -f "$ROOT_OUT/frozen_policy.json"
test -f "$ROOT_OUT/pilot_result.json"
echo "[done] dual-analyzer pilot codec=$CODEC"
