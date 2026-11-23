"""
build_mix.py  -  Steps 1-3 for Humnava OCR retraining

1. Checks train.tsv
2. Makes many augmented copies of each real line
3. Mixes them with augmented rendered Nastaliq lines

Output (in ~/humnava/mix_data):
    images/      all images
    labels.tsv   train labels  (filename<TAB>text, NORMAL reading order)
    val.tsv      validation labels (same format)

Run:
    cd ~/humnava && source .venv/bin/activate
    python build_mix.py
"""
import argparse
import io
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

HOME = Path.home() / "humnava"


def read_tsv(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2 or not parts[1].strip():
            print("  BAD LINE (skipped):", line[:60])
            continue
        rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def augment(img):
    img = img.convert("RGB")
    bg = img.getpixel((0, 0))
    # small tilt
    img = img.rotate(random.uniform(-2, 2), expand=True, fillcolor=bg,
                     resample=Image.BICUBIC)
    w, h = img.size
    # lower resolution
    s = random.uniform(0.5, 1.0)
    if s < 0.95:
        small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        img = small.resize((w, h), Image.BILINEAR)
    # blur
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    # brightness and contrast
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.2))
    # noise
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, random.uniform(2, 12), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # jpeg compression
    if random.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(35, 85))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", default=HOME / "train.tsv")
    ap.add_argument("--real_dir", default=HOME / "my_train_lines" / "train_good")
    ap.add_argument("--test_tsv", default=HOME / "photos.tsv")
    ap.add_argument("--rendered_dir", default=HOME / "nastaliq_data" / "images")
    ap.add_argument("--rendered_tsv", default=HOME / "nastaliq_data" / "labels.tsv")
    ap.add_argument("--out", default=HOME / "mix_data")
    ap.add_argument("--copies", type=int, default=20, help="copies per real line")
    ap.add_argument("--n_rendered", type=int, default=1200)
    ap.add_argument("--n_val_real", type=int, default=8)
    ap.add_argument("--n_val_rendered", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    real_dir, rendered_dir = Path(a.real_dir), Path(a.rendered_dir)

    # ---------- Step 1: check train.tsv ----------
    print("Step 1: checking train.tsv")
    rows = read_tsv(Path(a.train_tsv))
    test_texts = {t for _, t in read_tsv(Path(a.test_tsv))} if Path(a.test_tsv).exists() else set()

    good, seen_files, seen_texts = [], set(), set()
    for name, text in rows:
        if not (real_dir / name).exists():
            print("  MISSING IMAGE (skipped):", name)
        elif name in seen_files:
            print("  DUPLICATE FILE (skipped):", name)
        elif text in seen_texts:
            print("  DUPLICATE TEXT (skipped):", name)
        elif text in test_texts:
            print("  SAME TEXT AS A TEST LINE (skipped):", name)
        else:
            if re.search(r"[A-Za-z0-9]", text):
                print("  WARNING: English letters or digits in text:", name)
            good.append((name, text))
            seen_files.add(name)
            seen_texts.add(text)
    print(f"  {len(good)} good lines out of {len(rows)} rows")

    random.shuffle(good)
    val_real = good[:a.n_val_real]
    train_real = good[a.n_val_real:]

    # ---------- output folder ----------
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    train_rows, val_rows = [], []

    # ---------- Step 2: real lines ----------
    print("Step 2: augmenting real lines")
    for i, (name, text) in enumerate(train_real):
        img = Image.open(real_dir / name).convert("RGB")
        for k in range(a.copies):
            fn = f"real_{i:03d}_{k:02d}.png"
            (img if k == 0 else augment(img)).save(out / "images" / fn)
            train_rows.append((fn, text))
    for i, (name, text) in enumerate(val_real):
        fn = f"valreal_{i:03d}.png"
        Image.open(real_dir / name).convert("RGB").save(out / "images" / fn)
        val_rows.append((fn, text))
    print(f"  real train images: {len(train_rows)}, real val images: {len(val_rows)}")

    # ---------- Step 3: rendered lines ----------
    print("Step 3: mixing in rendered lines")
    rend = [(n, t) for n, t in read_tsv(Path(a.rendered_tsv)) if (rendered_dir / n).exists()]
    random.shuffle(rend)
    need = a.n_rendered + a.n_val_rendered
    rend = rend[:need]
    for i, (name, text) in enumerate(rend[:a.n_rendered]):
        fn = f"rend_{i:04d}.png"
        augment(Image.open(rendered_dir / name)).save(out / "images" / fn)
        train_rows.append((fn, text))
    for i, (name, text) in enumerate(rend[a.n_rendered:]):
        fn = f"valrend_{i:03d}.png"
        Image.open(rendered_dir / name).convert("RGB").save(out / "images" / fn)
        val_rows.append((fn, text))

    random.shuffle(train_rows)
    (out / "labels.tsv").write_text(
        "".join(f"{n}\t{t}\n" for n, t in train_rows), encoding="utf-8")
    (out / "val.tsv").write_text(
        "".join(f"{n}\t{t}\n" for n, t in val_rows), encoding="utf-8")

    print(f"\nDone. Train: {len(train_rows)} lines, Val: {len(val_rows)} lines")
    print("Saved in", out)


if __name__ == "__main__":
    main()
