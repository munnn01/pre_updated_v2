set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
REPO=/kaggle/working/test_pre
OUT=/kaggle/working/outputs/paper_ar_visual
mkdir -p "$OUT" /kaggle/working/paper_cache
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf paper_ar_visual.tgz outputs/paper_ar_visual 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/paper_ar_visual.tgz"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/test_pre.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision, cv2, PIL; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "visual-only CPU run")'
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then KIN_ROOT="$candidate"; break; fi
done
test -n "${KIN_ROOT:-}" || { echo 'Missing Kinetics dataset' >&2; exit 2; }
for candidate in /kaggle/input/v2-paper-cache-1000-20260924 /kaggle/input/datasets/qktttttttttt/v2-paper-cache-1000-20260924; do
  if [ -d "$candidate" ]; then CACHE_SOURCE="$candidate"; break; fi
done
test -n "${CACHE_SOURCE:-}" || { echo 'Missing private V2 cache dataset' >&2; exit 2; }
for codec in h264 h265; do
  if [ -d "$CACHE_SOURCE/$codec/$codec/shard_0" ]; then
    cp -r "$CACHE_SOURCE/$codec/$codec" /kaggle/working/paper_cache/
  elif [ -d "$CACHE_SOURCE/$codec/shard_0" ]; then
    cp -r "$CACHE_SOURCE/$codec" /kaggle/working/paper_cache/
  elif [ -f "$CACHE_SOURCE/$codec.zip" ]; then
    unzip -q "$CACHE_SOURCE/$codec.zip" -d /kaggle/working/paper_cache
  else
    echo "Missing cached $codec shard directories" >&2
    exit 2
  fi
  test -f "/kaggle/working/paper_cache/$codec/shard_0/shard_records.jsonl"
  test -f "/kaggle/working/paper_cache/$codec/shard_1/shard_records.jsonl"
done
python scripts/build_train_index.py --root "$KIN_ROOT" --out /kaggle/working/kinetics_hash_split.json --assert-fingerprint 30f083f8520a
for codec in h264 h265; do
  python ops/paper_ar_visual.py \
    --index /kaggle/working/kinetics_hash_split.json \
    --cache-dir "/kaggle/working/paper_cache/$codec/shard_0" \
    --cache-dir "/kaggle/working/paper_cache/$codec/shard_1" \
    --codec "$codec" --qp 40 --count 8 --out-dir "$OUT/$codec"
done
test -f "$OUT/h264/h264_visual_manifest.json"
test -f "$OUT/h265/h265_visual_manifest.json"
echo '[done] paired Kinetics panels for both codecs'
