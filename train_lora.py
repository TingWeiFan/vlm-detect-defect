"""
LoRA fine-tuning of Qwen3-VL for defect detection on a DatasetNinja (Supervisely) dataset.

Two tasks are trained jointly, matching exactly what detect_defect.py asks at inference:
  * cls : CLS_PROMPT     -> "Yes" / "No"                      (stage-1 gate)
  * det : DEFAULT_PROMPT -> [{"bbox_2d": [x1,y1,x2,y2], ...}] or []  (0..1000 relative coords)
GT boxes come from connected components of the GT masks.

The train split is divided into train / val (val is used later to calibrate the gate threshold);
the test split is never touched here.

Usage:
  python train_lora.py --dataset kolektorsdd2-DatasetNinja/train --out lora_qwen3vl2b
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from detect_defect import CLS_PROMPT, DEFAULT_PROMPT, apply_clahe, load_gt_mask, tile_windows

LORA_TARGETS = (r".*(language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
                r"|visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2)))")


def gt_boxes(mask, min_area=10):
    """Connected components of the GT mask -> boxes in Qwen3-VL 0..1000 relative coords."""
    H, W = mask.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, w, h, a in st[1:]:
        if a < min_area:
            continue
        boxes.append([round(x / W * 1000), round(y / H * 1000),
                      round((x + w) / W * 1000), round((y + h) / H * 1000)])
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def split_items(root, val_frac, seed):
    items = []
    for p in sorted((root / "img").glob("*")):
        ann = root / "ann" / (p.name + ".json")
        items.append((p, ann, '"bitmap"' in ann.read_text()))
    rng = random.Random(seed)
    val = set()
    for label in (True, False):  # stratified
        group = [p.name for p, _, d in items if d == label]
        val |= set(rng.sample(group, round(len(group) * val_frac)))
    return [it for it in items if it[0].name not in val], [it for it in items if it[0].name in val]


def expand_tiles(items):
    """(path, ann, has_defect) per image -> (path, ann, has_defect, window) per square tile.
    A tile is positive iff it contains a GT component (after clipping) that gt_boxes keeps."""
    out = []
    for p, ann, _ in items:
        W, H = Image.open(p).size
        mask = load_gt_mask(ann, H, W)
        for w in tile_windows(W, H):
            x0, y0, x1, y1 = w
            out.append((p, ann, bool(gt_boxes(mask[y0:y1, x0:x1])), w))
    return out


class Sample:
    def __init__(self, path, ann, task, flip_h, flip_v, window=None):
        self.path, self.ann, self.task, self.flip_h, self.flip_v = path, ann, task, flip_h, flip_v
        self.window = window  # (x0, y0, x1, y1) crop for tile training, None = whole image


def build_example(s: Sample, processor, upscale, clahe=0.0, cls_upscale=1.0):
    image = Image.open(s.path).convert("RGB")
    if clahe:
        image = apply_clahe(image, clahe)
    W, H = image.size
    mask = load_gt_mask(s.ann, H, W)
    if s.window:
        x0, y0, x1, y1 = s.window
        image, mask = image.crop(s.window), mask[y0:y1, x0:x1]
        W, H = image.size
    if s.flip_h:
        image, mask = image.transpose(Image.FLIP_LEFT_RIGHT), mask[:, ::-1]
    if s.flip_v:
        image, mask = image.transpose(Image.FLIP_TOP_BOTTOM), mask[::-1]
    boxes = gt_boxes(mask)
    if s.task == "cls":
        prompt, answer = CLS_PROMPT, "Yes" if boxes else "No"
        if cls_upscale != 1.0:
            image = image.resize((int(W * cls_upscale), int(H * cls_upscale)), Image.BICUBIC)
    else:
        image = image.resize((int(W * upscale), int(H * upscale)), Image.BICUBIC)
        prompt = DEFAULT_PROMPT
        answer = json.dumps([{"bbox_2d": b, "label": "defect"} for b in boxes])
    user = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    enc = processor.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt")
    n_prompt = processor.apply_chat_template(user, tokenize=True, add_generation_prompt=True,
                                             return_dict=True, return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100  # loss only on the answer (+ <|im_end|>)
    enc["labels"] = labels
    return enc


def make_epoch(train_items, pos_repeat, neg_ratio, rng):
    pos = [it for it in train_items if it[2]]
    neg = [it for it in train_items if not it[2]]
    chosen = pos * pos_repeat
    chosen += rng.sample(neg, min(len(neg), int(len(chosen) * neg_ratio)))
    samples = [Sample(it[0], it[1], task, rng.random() < 0.5, rng.random() < 0.5,
                      it[3] if len(it) > 3 else None)
               for it in chosen for task in ("cls", "det")]
    rng.shuffle(samples)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kolektorsdd2-DatasetNinja/train")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", default="lora_qwen3vl2b")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8, help="累積多少步更新梯度, 效果等同於batch size = 8")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--upscale", type=float, default=2.0, help="det-task upscale (use same at inference)")
    ap.add_argument("--cls-upscale", type=float, default=1.0, help="cls-task upscale (stored for inference)")
    ap.add_argument("--tile", type=int, default=0, help="1 = train on overlapping square tiles (stored)")
    ap.add_argument("--clahe", type=float, default=0.0, help="CLAHE clip limit (0=off); stored for inference")
    ap.add_argument("--pos-repeat", type=int, default=3, help="oversample defect images per epoch")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="normal images per defect sample")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_items, val_items = split_items(Path(args.dataset), args.val_frac, args.seed)
    json.dump({"train": [p.name for p, _, _ in train_items], "val": [p.name for p, _, _ in val_items]}, open(out / "split.json", "w"))
    print(f"train {len(train_items)} ({sum(d for *_, d in train_items)} defect) val {len(val_items)} ({sum(d for *_, d in val_items)} defect)")

    if args.tile:
        train_items = expand_tiles(train_items)
        print(f"tiles: {len(train_items)} ({sum(it[2] for it in train_items)} with defect)")

    # 用 bf16 載入模型，用梯度檢查點省記憶體，再把模型包成 LoRA，只訓練 1.1% 的參數。這樣才能在 10GB 的 RTX 3080 上微調一個 VLM。
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank,
                                             lora_alpha=2 * args.rank,
                                             lora_dropout=0.05,
                                             target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    # 先算出總共要更新 628 次；只把 LoRA 參數交給 AdamW；學習率前 31 步慢慢升到 1e-4，之後沿餘弦曲線降到 0
    steps_per_epoch = math.ceil(len(make_epoch(train_items, args.pos_repeat, args.neg_ratio, random.Random(0))) / args.grad_accum)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)

    model.train()
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        samples = make_epoch(train_items, args.pos_repeat, args.neg_ratio, rng)
        run = {"cls": [], "det": []}
        for i, s in enumerate(samples):
            batch = build_example(s, processor, args.upscale, args.clahe, args.cls_upscale).to("cuda")
            loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            run[s.task].append(loss.item())
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(samples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    el = time.time() - t0
                    print(f"ep {ep + 1} step {step}/{total}  loss cls {np.mean(run['cls'][-80:]):.4f} "
                          f"det {np.mean(run['det'][-80:]):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                          f"{el / 60:.1f}m elapsed, ETA {el / step * (total - step) / 60:.1f}m", flush=True)
        model.save_pretrained(out)
        json.dump(vars(args), open(out / "train_args.json", "w"), indent=1)
        print(f"epoch {ep + 1} done, adapter saved to {out}/", flush=True)


if __name__ == "__main__":
    main()
