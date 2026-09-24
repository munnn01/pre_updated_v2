set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
REF="__REF__"
REPO=/kaggle/working/pre_updated_v2
OUT=/kaggle/working/outputs/paper_coco_visual
mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf paper_coco_visual.tgz outputs/paper_coco_visual 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/paper_coco_visual.tgz"
  exit "$rc"
}
trap finish EXIT
git clone -q https://github.com/munnn01/pre_updated_v2.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision, cv2, pycocotools, PIL; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); assert torch.cuda.is_available()'
for candidate in /kaggle/input/coco-2017-dataset/coco2017 /kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017; do
  if [ -d "$candidate/val2017" ]; then COCO_ROOT="$candidate"; break; fi
done
test -n "${COCO_ROOT:-}" || { echo 'Missing COCO val2017 dataset' >&2; exit 2; }
python ops/probe_background_suppression.py \
  --images "$COCO_ROOT/val2017" \
  --ann "$COCO_ROOT/annotations/instances_val2017.json" \
  --n-images 100 --size 320 --seed 20260924 \
  --qps 35,40,45 --codecs h264,h265 --sigmas 4 \
  --roi-sigmas 0 --post-sigmas 0 \
  --visual-count 8 --visual-qp 40 --bootstrap 100 \
  --out "$OUT"
test -f "$OUT/probe_bgsuppress.json"
test -f "$OUT/visuals/visual_manifest.json"
echo '[done] paired COCO panels and OD mAP curves for both codecs'
