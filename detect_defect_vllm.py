"""
vLLM backend for the Qwen3-VL + SAM2 defect pipeline (same logic as detect_defect.py, faster).

Requires a *merged* model (LoRA folded into the base weights, see README / merge step), because
our LoRA also adapts the vision tower, which vLLM's LoRA support does not cover.
Preprocessing settings (tile / upscale / cls_upscale / clahe) are read from <model>/train_args.json.

All Qwen requests of a batch of images (every tile's Yes/No gate, then every passing tile's box
generation) are sent to vLLM at once so they run batched on the GPU.
Stage 3 (default on, --no-describe to skip): every final box is cropped with context and the same
model is asked for {type, color, shape, description}; "label" becomes the type.


Usage (conda env: vllm):
  python detect_defect_vllm.py --model merged_qwen3vl2b_tile --dataset kolektorsdd2-DatasetNinja/test \
      --cls-threshold 0.963 --out results_test_vllm
  python detect_defect_vllm.py --model merged_qwen3vl2b_tile --images a.png b.png --cls-threshold 0.963
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from detect_defect import (CLS_PROMPT, DEFAULT_PROMPT, DESC_PROMPT, Sam2Segmenter, apply_clahe,
                           apply_description, desc_crop, draw, load_gt_mask, mask_to_polygons, measure_mask,
                           merge_dets, parse_json_list, size_text, tile_windows)


class VllmQwen:
    def __init__(self, model_dir, gpu_mem=0.75, max_model_len=3072, profile_px=1024):
        targs = json.load(open(Path(model_dir) / "train_args.json"))
        self.upscale = targs.get("upscale", 3.0)
        self.cls_upscale = targs.get("cls_upscale", 1.0)
        self.clahe = targs.get("clahe", 0.0)
        self.tile = bool(targs.get("tile", 0))

        # 載入模型的文字工具，並找出「Yes」和「No」的編號
        self.processor = AutoProcessor.from_pretrained(model_dir)
        tok = self.processor.tokenizer
        self.yes_id, self.no_id = tok.encode("Yes")[0], tok.encode("No")[0]
        
        # 用 vLLM 把模型載進 GPU，建立推論引擎
        self.llm = LLM(model=model_dir,
                       dtype="bfloat16",
                       gpu_memory_utilization=gpu_mem,
                       max_model_len=max_model_len,
                       max_num_seqs=16,
                       limit_mm_per_prompt={"image": {"count": 1, "width": profile_px, "height": profile_px}},
                       seed=0)
        
        self.p_cls = self._prompt(CLS_PROMPT)
        self.p_det = self._prompt(DEFAULT_PROMPT)
        
        # 設定模型生成時的規則
        self.sp_cls = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        self.sp_det = SamplingParams(max_tokens=512, temperature=0.0)

        self.p_desc = self._prompt(DESC_PROMPT)
        self.sp_desc = SamplingParams(max_tokens=150, temperature=0.0)

    # 轉成標準格式字串
    def _prompt(self, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def windows(self, image):
        W, H = image.size
        return tile_windows(W, H) if self.tile else [(0, 0, W, H)]

    def _p_yes(self, out):
        # 取出第一個 token 的候選名單
        lp = out.outputs[0].logprobs[0]

        # 查 Yes 和 No 的 log 機率
        y = lp[self.yes_id].logprob if self.yes_id in lp else -math.inf
        n = lp[self.no_id].logprob if self.no_id in lp else -math.inf
        if y == n == -math.inf:
            return 0.0
        m = max(y, n)
        return math.exp(y - m) / (math.exp(y - m) + math.exp(n - m))

    @staticmethod
    def _resize(img, f):
        return img if f == 1.0 else img.resize((int(img.width * f), int(img.height * f)), Image.BICUBIC)

    def describe(self, images, dets_per_image, timing=None):
        """Stage 3: one batched request per final box (crop around it) -> adds type/color/shape/description."""
        jobs = [(i, d) for i, dets in enumerate(dets_per_image) for d in dets]
        if not jobs:
            return
        t0 = time.perf_counter()
        outs = self.llm.generate(
            [{"prompt": self.p_desc, "multi_modal_data": {"image": desc_crop(images[i], d["bbox"])}}
             for i, d in jobs], self.sp_desc, use_tqdm=False)
        for (_, d), o in zip(jobs, outs):
            apply_description(d, o.outputs[0].text)
        if timing is not None:
            timing["desc"] += time.perf_counter() - t0

    def run(self, images, thr, timing=None, describe=True):
        """images: list of PIL. Returns [(score, dets, raw)] per image, in full-image coordinates."""
        crops = []  # (image index, window, crop)
        for i, im in enumerate(images):
            src = apply_clahe(im, self.clahe) if self.clahe else im
            # 把一張圖切成數個互相重疊的正方形小塊
            for w in self.windows(im):
                crops.append((i, w, src.crop(w)))

        t0 = time.perf_counter()
        if thr > 0:
            outs = self.llm.generate(
                [{"prompt": self.p_cls, "multi_modal_data": {"image": self._resize(c, self.cls_upscale)}}
                 for _, _, c in crops], self.sp_cls, use_tqdm=False)
            scores = [self._p_yes(o) for o in outs]
        else:
            scores = [1.0] * len(crops)
        t1 = time.perf_counter()

        passing = [k for k, s in enumerate(scores) if thr <= 0 or s >= thr]
        texts = {}
        if passing:
            outs = self.llm.generate(
                [{"prompt": self.p_det, "multi_modal_data": {"image": self._resize(crops[k][2], self.upscale)}}
                 for k in passing], self.sp_det, use_tqdm=False)
            texts = {k: o.outputs[0].text for k, o in zip(passing, outs)}
        t2 = time.perf_counter()
        if timing is not None:
            timing["gate"] += t1 - t0
            timing["det"] += t2 - t1

        results = [[[], [], []] for _ in images]  # scores, dets, raws
        for k, (i, (x0, y0, x1, y1), c) in enumerate(crops):
            results[i][0].append(scores[k])
            if k not in texts:
                continue
            results[i][2].append(f"tile({x0},{y0}) score={scores[k]:.3f}: {texts[k]}")
            Wt, Ht = c.size  # Qwen3-VL boxes are relative 0..1000 on the crop
            for d in parse_json_list(texts[k]):
                b = d.get("bbox_2d") or d.get("bbox")
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    continue
                bx1, bx2 = sorted((float(b[0]) * Wt / 1000, float(b[2]) * Wt / 1000))
                by1, by2 = sorted((float(b[1]) * Ht / 1000, float(b[3]) * Ht / 1000))
                bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(Wt - 1, bx2), min(Ht - 1, by2)
                if bx2 - bx1 < 2 or by2 - by1 < 2:
                    continue
                results[i][1].append({"bbox": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
                                      "label": str(d.get("label", "defect"))})
        
        # 還原 image
        merged = [merge_dets(d) for _, d, _ in results]
        if describe:
            self.describe(images, merged, timing)
        return [(max(s), d, "\n".join(r) or "(all tiles rejected by gate)")
                for (s, _, r), d in zip(results, merged)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir (contains train_args.json)")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--dataset")
    ap.add_argument("--ids-file", help="一個 JSON 檔，只保留裡面列出的檔名，用來挑出 train/val/test 子集")
    ap.add_argument("--ids-key", default="val", help="從 --ids-file 的哪個 key 取檔名清單")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張")
    ap.add_argument("--out", default="results_vllm")
    ap.add_argument("--cls-threshold", type=float, required=True)
    ap.add_argument("--batch-images", type=int, default=1, help="images sent to vLLM together (1 = per-image latency; larger = throughput)")
    ap.add_argument("--gpu-mem", type=float, default=0.75, help="vLLM share of GPU memory (rest: SAM2, display)")
    ap.add_argument("--max-model-len", type=int, default=3072, help="vLLM 單一序列的最大長度，包含 image tokens、prompt 和輸出")
    ap.add_argument("--profile-px", type=int, default=1024, help="vLLM 啟動時用多大的圖去估算記憶體")
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--mm-per-px", type=float, default=0.0,
                    help="physical pixel size; if set, defect sizes are also reported in mm")
    ap.add_argument("--no-describe", action="store_true", help="skip stage 3 (defect type/description)")
    args = ap.parse_args()

    items = [(Path(p), None) for p in args.images]
    if args.dataset:
        root = Path(args.dataset)
        items += [(p, root / "ann" / (p.name + ".json")) for p in sorted((root / "img").glob("*"))]
    if args.ids_file:
        keep = set(json.load(open(args.ids_file))[args.ids_key])
        items = [it for it in items if it[0].name in keep]
    if args.limit:
        items = items[: args.limit]

    qwen = VllmQwen(args.model, gpu_mem=args.gpu_mem, max_model_len=args.max_model_len, profile_px=args.profile_px)
    sam = Sam2Segmenter(args.sam)
    print(f"tile={qwen.tile} upscale={qwen.upscale} cls_upscale={qwen.cls_upscale} clahe={qwen.clahe}")
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # warm-up (CUDA graphs, image encoder) so timings are steady-state
    warm = Image.open(items[0][0]).convert("RGB")
    for _ in range(2):
        qwen.run([warm], 0.0, describe=not args.no_describe)
        sam.segment(warm, [[10, 10, 60, 60]])

    stats = dict(tp=0, fp=0, fn=0, tn=0, inter=0, union=0)
    records, lat = [], []
    timing = {"gate": 0.0, "det": 0.0, "desc": 0.0, "sam": 0.0}
    
    t_all = time.perf_counter()
    for b0 in range(0, len(items), args.batch_images):
        batch = items[b0:b0 + args.batch_images]
        images = [Image.open(p).convert("RGB") for p, _ in batch]

        t0 = time.perf_counter()
        outs = qwen.run(images, args.cls_threshold, timing, describe=not args.no_describe)
        for (p, ann), image, (score, dets, raw) in zip(batch, images, outs):
            ts = time.perf_counter()
            masks = sam.segment(image, [d["bbox"] for d in dets])
            torch.cuda.synchronize() # 讓 CPU 停下來，等 GPU 把手上所有工作做完再繼續. For sam2 calculating
            timing["sam"] += time.perf_counter() - ts
            for d, m in zip(dets, masks):
                d["size"] = measure_mask(m, args.mm_per_px)
            
            W, H = image.size
            pred = np.zeros((H, W), bool)
            for m in masks:
                pred |= m # 把這張遮罩「疊上去」

            gt = load_gt_mask(ann, H, W) if ann else None
            rec = {"image": str(p), "defect_score": round(score, 4), "qwen_raw": raw,
                   "defects": [{**{k: v for k, v in d.items() if k != "bbox"},
                                "bbox": [round(v, 1) for v in d["bbox"]],
                                "contours": mask_to_polygons(m), "area": int(m.sum())}
                               for d, m in zip(dets, masks)]}
            msg = f"[{len(records) + 1}/{len(items)}] {p.name}: score={score:.3f} {len(dets)} defect(s)"
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
            if not args.no_vis:
                cv2.imwrite(str(out / f"{p.stem}_vis.png"), cv2.cvtColor(draw(np.array(image), dets, masks, gt), cv2.COLOR_RGB2BGR))
        lat.append((time.perf_counter() - t0) / len(batch))
    wall = time.perf_counter() - t_all

    json.dump(records, open(out / "results.json", "w"), ensure_ascii=False, indent=1)
    if any(r.get("iou") is not None for r in records):
        s = stats
        print(f"\nImage-level: TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}  "
              f"precision={s['tp'] / max(1, s['tp'] + s['fp']):.3f} recall={s['tp'] / max(1, s['tp'] + s['fn']):.3f}")
        print(f"Pixel-level IoU (dataset): {s['inter'] / max(1, s['union']):.3f}")
    n = len(items)
    print(f"Timing over {n} images (batch-images={args.batch_images}, incl. visualization I/O): "
          f"wall {wall:.1f}s = {wall / n * 1000:.0f} ms/image | Qwen gate {timing['gate'] / n * 1000:.0f} + "
          f"det {timing['det'] / n * 1000:.0f} + desc {timing['desc'] / n * 1000:.0f} + SAM {timing['sam'] / n * 1000:.0f} ms/image (avg)")
    if not args.no_describe:
        from collections import Counter
        c = Counter(d["label"] for r in records for d in r["defects"])
        print("Defect types:", ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
