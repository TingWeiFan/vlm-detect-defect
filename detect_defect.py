"""
Qwen-VL + SAM2 defect contour detection.

Pipeline:
  1. Qwen-VL (Qwen3-VL / Qwen2.5-VL) looks at the image and returns defect bounding boxes (JSON).
  2. Each box is used as a prompt for SAM2, which returns a pixel mask.
  3. (default on, --no-describe to skip) each box is cropped with context and Qwen is asked for
     {type, color, shape, description}; "label" becomes the type.
  4. cv2.findContours turns the mask into a contour polygon; length / width / area are measured
     from the mask (rotated min-area rectangle), in px or mm with --mm-per-px.

Usage:
  python detect_defect.py --images path/to/img.png [more.png ...] --out results/
  python detect_defect.py --dataset kolektorsdd2-DatasetNinja/test --only-defect --limit 20 --out results/
"""
import argparse
import base64
import io
import json
import re
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Sam2Model, Sam2Processor

DEFAULT_PROMPT = (
    "This is a close-up image of an industrial product surface used for quality inspection. "
    "Carefully look for any surface defects such as scratches, spots, stains, dents, cracks, "
    "pits or discoloration. Normal texture noise is NOT a defect. "
    "Output every defect as a JSON list: "
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<defect type>"}]. '
    "If there is no defect, output []."
)
# Stage-1 gate: Yes/No question; P(Yes) is used as an image-level anomaly score.
CLS_PROMPT = ("Is there an obvious anomaly on this surface that differs clearly from the "
              "uniform background texture? Answer Yes or No.")

# Stage-3 description of each detected box (zero-shot; the LoRA was never trained on this prompt).
DESC_TYPES = ["scratch", "crack", "dent", "pit", "stain", "spot", "discoloration", "contamination", "other"]
DESC_PROMPT = ("This is a close-up crop of a defect on a metal/plastic commutator surface, centered in the image. "
               'Describe it as JSON: {"type": one of [' + ", ".join(DESC_TYPES) + '], '
               '"color": short phrase, "shape": short phrase, "description": one sentence}')


def parse_json_dict(text):
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", text), re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return d if isinstance(d, dict) else {}


def desc_crop(image, bbox, side=448):
    """Square crop centred on the box with context (2.5x box + 40 px), resized to `side`."""
    W, H = image.size
    x1, y1, x2, y2 = bbox
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 2.5 + 40
    c = image.crop((max(0, cx - s / 2), max(0, cy - s / 2), min(W, cx + s / 2), min(H, cy + s / 2)))
    f = side / max(c.size)
    return c.resize((max(1, round(c.width * f)), max(1, round(c.height * f))), Image.BICUBIC)


def apply_description(det, text):
    """Fill type/color/shape/description of one detection from the model's answer."""
    info = parse_json_dict(text)
    t = str(info.get("type", "")).lower().strip()
    det["label"] = t if t in DESC_TYPES else ("other" if t else "defect")
    for k in ("color", "shape", "description"):
        if info.get(k):
            det[k] = str(info[k])
    if not info:
        det["desc_raw"] = text


def apply_clahe(image: Image.Image, clip=2.0, tile=8) -> Image.Image:
    """Local contrast enhancement on the L channel (LAB); makes faint defects more visible."""
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# ----------------------------------------------------------------------------- Qwen-VL
class QwenDetector:
    def __init__(self, model_id, load_4bit=False, upscale=3.0, max_new_tokens=512, lora=None,
                 clahe=0.0, cls_upscale=1.0):
        kw = dict(dtype=torch.bfloat16, device_map="cuda")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora).merge_and_unload().eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Qwen3-VL outputs coordinates normalized to 0..1000; Qwen2.5-VL outputs absolute
        # pixel coordinates on the (resized) image fed to the model.
        self.relative_coords = "qwen3" in model_id.lower()
        self.upscale = upscale
        self.clahe = clahe  # CLAHE clip limit applied to the image fed to Qwen; 0 = off
        self.cls_upscale = cls_upscale  # resize factor for the Yes/No gate
        self.max_new_tokens = max_new_tokens
        tok = self.processor.tokenizer
        self.yes_id, self.no_id = tok.encode("Yes")[0], tok.encode("No")[0]

    @torch.inference_mode()
    def defect_score(self, image: Image.Image, prompt: str = CLS_PROMPT) -> float:
        """P(Yes) for 'is there a defect?' -- used to reject defect-free images."""
        if self.clahe:
            image = apply_clahe(image, self.clahe)
        if self.cls_upscale != 1.0:
            W, H = image.size
            image = image.resize((int(W * self.cls_upscale), int(H * self.cls_upscale)), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1].float()
        return torch.softmax(logits[[self.yes_id, self.no_id]], 0)[0].item()

    @torch.inference_mode()
    def detect(self, image: Image.Image, prompt: str):
        W, H = image.size
        # Small, low-contrast defects are easier for the VLM to see on an enlarged image.
        if self.clahe:
            image = apply_clahe(image, self.clahe)
        big = image.resize((int(W * self.upscale), int(H * self.upscale)), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": big},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]

        if self.relative_coords:
            sx, sy = W / 1000.0, H / 1000.0
        else:
            # size of the image as actually seen by the model (after smart_resize)
            _, gh, gw = inputs["image_grid_thw"][0].tolist()
            patch = self.processor.image_processor.patch_size
            sx, sy = W / (gw * patch), H / (gh * patch)

        dets = []
        for d in parse_json_list(text):
            b = d.get("bbox_2d") or d.get("bbox")
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            x1, x2 = sorted((x1 * sx, x2 * sx))
            y1, y2 = sorted((y1 * sy, y2 * sy))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            dets.append({"bbox": [x1, y1, x2, y2], "label": str(d.get("label", "defect"))})
        return dets, text

    @torch.inference_mode()
    def describe(self, image: Image.Image, dets):
        """Stage 3: ask for type/color/shape/description of every box (crop around it), in place."""
        for d in dets:
            messages = [{"role": "user", "content": [{"type": "image", "image": desc_crop(image, d["bbox"])},
                                                     {"type": "text", "text": DESC_PROMPT}]}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
            apply_description(d, self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                                             skip_special_tokens=True)[0])


def parse_json_list(text):
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        # fall back: grab every [x1,y1,x2,y2] quadruple
        return [{"bbox_2d": [float(v) for v in q]} for q in
                re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)]
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


# ----------------------------------------------------------------------------- SAM2
class Sam2Segmenter:
    def __init__(self, model_id="facebook/sam2.1-hiera-small"):
        self.model = Sam2Model.from_pretrained(model_id, dtype=torch.float32).to("cuda").eval()
        self.processor = Sam2Processor.from_pretrained(model_id)

    @torch.inference_mode()
    def segment(self, image: Image.Image, boxes):
        if not boxes:
            return []
        inputs = self.processor(images=image, input_boxes=[[list(b) for b in boxes]],
                                return_tensors="pt").to("cuda")
        out = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        # masks: (num_boxes, 1, H, W)
        masks = masks[:, 0].numpy().astype(bool)
        # keep only the part of the mask inside (a slightly enlarged) box to avoid leaking
        H, W = masks.shape[1:]
        res = []
        for m, (x1, y1, x2, y2) in zip(masks, boxes):
            pad = 0.15 * max(x2 - x1, y2 - y1)
            clip = np.zeros_like(m)
            clip[int(max(0, y1 - pad)):int(min(H, y2 + pad + 1)),
                 int(max(0, x1 - pad)):int(min(W, x2 + pad + 1))] = True
            res.append(m & clip)
        return res


# ----------------------------------------------------------------------------- utils
def mask_to_polygons(mask, min_area=4):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).tolist() for c in cnts if cv2.contourArea(c) >= min_area]


def measure_mask(mask, mm_per_px=0.0):
    """Geometric size of a defect from its SAM2 mask (all components together).
    length/width = sides of the minimum-area rotated rectangle around the defect pixels,
    max_extent = largest distance between two defect pixels (Feret diameter)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    w, h = w + 1, h + 1  # pixel centres -> pixel extent
    length, width = max(w, h), min(w, h)
    angle = ang if w >= h else ang + 90  # direction of the long side, degrees from the x axis
    hull = cv2.convexHull(pts).reshape(-1, 2)
    d = np.sqrt(((hull[:, None] - hull[None]) ** 2).sum(-1)).max() + 1
    size = {"length_px": round(float(length), 1), "width_px": round(float(width), 1),
            "max_extent_px": round(float(d), 1), "area_px": int(len(xs)),
            "aspect_ratio": round(float(length / width), 2), "angle_deg": round(float(angle % 180), 1)}
    if mm_per_px > 0:
        size.update(length_mm=round(length * mm_per_px, 3), width_mm=round(width * mm_per_px, 3),
                    max_extent_mm=round(d * mm_per_px, 3), area_mm2=round(len(xs) * mm_per_px ** 2, 4))
    return size


def size_text(size):
    if not size:
        return ""
    if "length_mm" in size:
        return f"{size['length_mm']:.2f}x{size['width_mm']:.2f}mm"
    return f"{size['length_px']:.0f}x{size['width_px']:.0f}px"


def load_gt_mask(ann_path, H, W):
    """Decode Supervisely bitmap annotations (DatasetNinja format) into a binary mask."""
    gt = np.zeros((H, W), bool)
    for o in json.load(open(ann_path)).get("objects", []):
        if o.get("geometryType") != "bitmap":
            continue
        m = np.array(Image.open(io.BytesIO(zlib.decompress(base64.b64decode(o["bitmap"]["data"])))))
        m = m[..., 3] if m.ndim == 3 else m
        x, y = o["bitmap"]["origin"]
        h, w = m.shape
        gt[y:y + h, x:x + w] |= m[: H - y, : W - x] > 0
    return gt


def draw(image_np, dets, masks, gt=None):
    vis = image_np.copy()
    if gt is not None and gt.any():  # GT contour in green
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 1)
    for d, m in zip(dets, masks):
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Qwen box: yellow
        overlay = vis.copy()
        overlay[m] = (255, 0, 0)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (255, 0, 0), 2)  # predicted contour: red
        cv2.putText(vis, (d["label"][:14] + " " + size_text(d.get("size"))).strip(), (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 0), 1, cv2.LINE_AA)
    return np.hstack([image_np, vis])


# ----------------------------------------------------------------------------- tiling
def tile_windows(W, H, stride_frac=0.7):
    """Square tiles of side min(W, H), overlapping by 1 - stride_frac, covering the whole image."""
    T = min(W, H)
    stride = max(1, int(T * stride_frac))

    def starts(L):
        s = list(range(0, L - T + 1, stride))
        if s[-1] != L - T:
            s.append(L - T)
        return s
    return [(x, y, x + T, y + T) for y in starts(H) for x in starts(W)]


def merge_dets(dets, thr=0.3):
    """Merge boxes found twice in overlapping tiles (intersection / smaller area > thr)."""
    dets = [dict(d) for d in dets]
    merged = True
    while merged:
        merged = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i]["bbox"], dets[j]["bbox"]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                if iw <= 0 or ih <= 0:
                    continue
                small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if iw * ih / max(small, 1e-6) > thr:
                    dets[i]["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del dets[j]
                    merged = True
                    break
            if merged:
                break
    return dets


def run_qwen(qwen, image, prompt, thr, tile):
    """Stage 1 + 2. Returns (image-level score, detections in full-image coords, raw text)."""
    if not tile:
        score = qwen.defect_score(image) if thr > 0 else 1.0
        if thr <= 0 or score >= thr:
            dets, raw = qwen.detect(image, prompt)
            return score, dets, raw
        return score, [], "(rejected by gate)"
    scores, dets, raws = [], [], []
    for x0, y0, x1, y1 in tile_windows(*image.size):
        crop = image.crop((x0, y0, x1, y1))
        s = qwen.defect_score(crop) if thr > 0 else 1.0
        scores.append(s)
        if thr <= 0 or s >= thr:
            d, r = qwen.detect(crop, prompt)
            raws.append(f"tile({x0},{y0}) score={s:.3f}: {r}")
            for dd in d:
                b = dd["bbox"]
                dets.append({**dd, "bbox": [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]})
    return max(scores), merge_dets(dets), "\n".join(raws) or "(all tiles rejected by gate)"


def image_score(qwen, image, tile):
    if not tile:
        return qwen.defect_score(image)
    return max(qwen.defect_score(image.crop(w)) for w in tile_windows(*image.size))


def calibrate(qwen, items, n, seed=0, tile=False):
    import random
    pos = [p for p, a in items if a and '"bitmap"' in a.read_text()]
    neg = [p for p, a in items if a and '"bitmap"' not in a.read_text()]
    random.seed(seed)
    pos, neg = random.sample(pos, min(n, len(pos))), random.sample(neg, min(n, len(neg)))
    s = np.array([image_score(qwen, Image.open(p).convert("RGB"), tile) for p in pos + neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    auc = (s[y == 1][:, None] > s[y == 0][None]).mean()
    # threshold maximizing balanced accuracy
    best = max(np.unique(s), key=lambda t: ((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) / 2)
    tpr, tnr = (s[y == 1] >= best).mean(), (s[y == 0] < best).mean()
    print(f"AUROC={auc:.3f}  best threshold={best:.3f}  recall={tpr:.3f}  normal-reject={tnr:.3f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--dataset", help="DatasetNinja split dir containing img/ and ann/")
    ap.add_argument("--only-defect", action="store_true", help="only images whose GT has defects")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--qwen", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--load-4bit", action="store_true", help="for 4B/8B models on 10GB GPUs")
    ap.add_argument("--lora", help="LoRA adapter dir from train_lora.py")
    ap.add_argument("--ids-file", help="split.json from train_lora.py; use with --ids-key")
    ap.add_argument("--ids-key", default="val")
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--upscale", type=float, default=None,
                    help="default: value from <lora>/train_args.json, else 3.0")
    ap.add_argument("--cls-upscale", type=float, default=None,
                    help="resize factor for the Yes/No gate; default: from <lora>/train_args.json, else 1.0")
    ap.add_argument("--clahe", type=float, default=None,
                    help="CLAHE clip limit for Qwen input (0=off); default: from <lora>/train_args.json, else 0")
    ap.add_argument("--tile", type=int, default=None,
                    help="1 = run Qwen on overlapping square tiles; default: from <lora>/train_args.json, else 0")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--cls-threshold", type=float, default=0.62,
                    help="P(Yes) gate; images below it are reported defect-free. <=0 disables")
    ap.add_argument("--mm-per-px", type=float, default=0.0,
                    help="physical pixel size; if set, defect sizes are also reported in mm")
    ap.add_argument("--no-describe", action="store_true", help="skip stage 3 (defect type/description)")
    ap.add_argument("--calibrate", type=int, default=0, metavar="N",
                    help="score N defect + N normal images of --dataset, print best threshold, exit")
    args = ap.parse_args()

    items = [(Path(p), None) for p in args.images]
    if args.dataset:
        root = Path(args.dataset)
        for p in sorted((root / "img").glob("*")):
            ann = root / "ann" / (p.name + ".json")
            if args.only_defect and '"bitmap"' not in ann.read_text():
                continue
            items.append((p, ann if ann.exists() else None))
    if args.ids_file:
        keep = set(json.load(open(args.ids_file))[args.ids_key])
        items = [it for it in items if it[0].name in keep]
    if args.limit:
        items = items[: args.limit]
    if not items:
        ap.error("no input images")

    # take preprocessing settings from the adapter's training config so train/infer always match
    targs = {}
    if args.lora and (Path(args.lora) / "train_args.json").exists():
        targs = json.load(open(Path(args.lora) / "train_args.json"))
    if args.upscale is None:
        args.upscale = targs.get("upscale", 3.0)
    if args.clahe is None:
        args.clahe = targs.get("clahe", 0.0)
    if args.cls_upscale is None:
        args.cls_upscale = targs.get("cls_upscale", 1.0)
    if args.tile is None:
        args.tile = int(targs.get("tile", 0))
    print(f"upscale={args.upscale} cls_upscale={args.cls_upscale} clahe={args.clahe} tile={args.tile}")
    qwen = QwenDetector(args.qwen, args.load_4bit, args.upscale, lora=args.lora, clahe=args.clahe,
                        cls_upscale=args.cls_upscale)
    if args.calibrate:
        calibrate(qwen, items, args.calibrate, tile=bool(args.tile))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sam = Sam2Segmenter(args.sam)

    stats = dict(tp=0, fp=0, fn=0, tn=0, inter=0, union=0)
    records = []
    for i, (p, ann) in enumerate(items):
        image = Image.open(p).convert("RGB")
        W, H = image.size
        score, dets, raw = run_qwen(qwen, image, args.prompt, args.cls_threshold, bool(args.tile))
        if not args.no_describe:
            qwen.describe(image, dets)
        masks = sam.segment(image, [d["bbox"] for d in dets])
        for d, m in zip(dets, masks):
            d["size"] = measure_mask(m, args.mm_per_px)
        pred = np.zeros((H, W), bool)
        for m in masks:
            pred |= m

        gt = load_gt_mask(ann, H, W) if ann else None
        rec = {"image": str(p), "defect_score": round(score, 4), "qwen_raw": raw,
               "defects": [{**{k: v for k, v in d.items() if k != "bbox"},
                            "bbox": [round(v, 1) for v in d["bbox"]],
                            "contours": mask_to_polygons(m), "area": int(m.sum())}
                           for d, m in zip(dets, masks)]}
        msg = f"[{i + 1}/{len(items)}] {p.name}: score={score:.3f} {len(dets)} defect(s)"
        if dets and not args.no_describe:
            msg += " [" + ", ".join(f"{d['label']} {size_text(d['size'])}" for d in dets) + "]"
        if gt is not None:
            has_gt, has_pred = gt.any(), pred.any()
            stats["tp" if has_gt and has_pred else "fn" if has_gt else "fp" if has_pred else "tn"] += 1
            inter, union = int((gt & pred).sum()), int((gt | pred).sum())
            stats["inter"] += inter
            stats["union"] += union
            rec["iou"] = inter / union if union else 1.0
            msg += f"  GT={'defect' if has_gt else 'ok'}  IoU={rec['iou']:.3f}"
        print(msg, flush=True)
        records.append(rec)
        cv2.imwrite(str(out / f"{p.stem}_vis.png"),
                    cv2.cvtColor(draw(np.array(image), dets, masks, gt), cv2.COLOR_RGB2BGR))

    json.dump(records, open(out / "results.json", "w"), ensure_ascii=False, indent=1)
    if any(r.get("iou") is not None for r in records):
        s = stats
        rec_ = s["tp"] / max(1, s["tp"] + s["fn"])
        prec = s["tp"] / max(1, s["tp"] + s["fp"])
        print(f"\nImage-level: TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}  "
              f"precision={prec:.3f} recall={rec_:.3f}")
        print(f"Pixel-level IoU (dataset): {s['inter'] / max(1, s['union']):.3f}")
    if not args.no_describe:
        from collections import Counter
        c = Counter(d["label"] for r in records for d in r["defects"])
        print("Defect types:", ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print(f"Saved visualizations + results.json to {out}/")


if __name__ == "__main__":
    main()
