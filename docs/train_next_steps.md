# Training Improvement Plan

## What was done

### Dataset fixes (already applied to code)

1. **Removed `suv` class** — had zero samples in VisDrone data. Renumbered to 6 classes.
2. **Renamed `compact_car` → `car`** — clearer label, matches VisDrone MOT `car` (category 4).
3. **Updated class mapping** in `_visdrone_constants.py` (VisDrone MOT category IDs):
   - `4: car → 0`, `5: van → 1`, `6: truck → 2`, `7/8: rickshaw → 3`, `9: bus → 4`, `10: motorcycle → 5`
4. **Added `--max-per-class` undersampling** to the converter script.
5. **Increased `cls` loss weight** (`0.5 → 2.0`) and **added focal loss** (`fl_gamma: 2.0`) to penalize misclassifying rare classes.
6. **Lowered learning rate** (`0.01 → 0.001`), **reduced aggressive augmentation** (`scale 0.5→0.3`, `mixup 0.1→0.0`, `copy_paste 0.1→0.0`, `degrees 5→3`), **increased patience** (`50 → 80`).

---

## What you need to do

### Step 1 — Re-run dataset conversion with undersampling

Cap van at ~60k to match the next largest class:

```bash
python scripts/convert_visdrone_mot_to_yolo.py \
    --input-dir <your_visdrone_path> \
    --output-dir data/vehicle_dataset \
    --copy-images \
    --max-per-class 60000
```

Expected resulting distribution:

| Class   | Before     | After (capped) |
|---------|------------|----------------|
| car     | 39,457     | 39,457         |
| van     | 501,184    | **60,000**     |
| truck   | 46,294     | 46,294         |
| rickshaw| 56,910     | 56,910         |
| bus     | 12,339     | 12,339         |
| motorcycle | 9,653  | 9,653          |

### Step 2 — Verify class distribution

```bash
for cls in 0 1 2 3 4 5; do
    echo -n "class $cls: "
    rg -c "^$cls " data/vehicle_dataset/labels/train/ | awk -F: '{s+=$2} END {print s}'
done
```

If `motorcycle` or `bus` is below 5k, consider merging or dropping them.

### Step 3 — Check annotation quality

`car` (formerly `compact_car`) had `mAP50 = 0.027` and `bus` had `mAP50 = 0.008` in the last training. This usually means:
- Labels are wrong or inconsistent
- Objects are too small to detect
- Class is ambiguous from a drone perspective

Visually inspect a few samples:

```bash
python scripts/inference.py \
    --source data/vehicle_dataset/images/val/ \
    --weights yolo26s.pt \
    --show
```

## Optional: Scale Up

If metrics are good and you want to push further:

1. **Fine-tune from best checkpoint**:
   ```yaml
   model: runs/train/yolo26_vehicle/weights/best.pt
   ```

2. **Try larger model**:
   ```yaml
   model: yolo26m.pt
   ```

3. **If undersampling helped but van recall dropped**, try `--max-per-class 80000` or `100000` to find the sweet spot.
