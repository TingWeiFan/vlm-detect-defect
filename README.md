# 以 VLM 進行金屬表面缺陷檢測

以 LoRA 微調 Qwen3-VL，只用 209 張缺陷影像的框標註訓練，在 [KolektorSDD2](https://datasetninja.com/kolektor-surface-defect-dataset-2) 上判斷影像有沒有缺陷並標出位置，再用 SAM2 切出缺陷遮罩，並比較 transformers 與 vLLM 兩種推論方式的速度。

## 與傳統模型比較

KolektorSDD2 官方 test set（1004 張，其中 110 張有缺陷），RTX 3080。

| Method | Input | Backend | Annotation | AP | Precision | Recall | Latency (ms) |
|:---|:---|:---|:---|---:|---:|---:|---:|
| Božič et al. [1] | full image | — | 246 masks | 0.954 | — | — | ~12 |
| DSR [2] | full image | — | 246 masks | 0.952 | — | — | ~17 |
| Ours | full image | transformers | 209 boxes | 0.974 | 0.935 | 0.918 | 449 |
| Ours | full image | vLLM | 209 boxes | 0.974 | 0.935 | 0.918 | 130 |
| Ours | 4 tiles | transformers | 209 boxes | 0.973 | 0.803 | 0.964 | ~1300 |
| Ours | 4 tiles | vLLM | 209 boxes | 0.973 | 0.803 | 0.964 | 712 |

## 參考資料

[1] J. Božič, D. Tabernik, D. Skočaj. *Mixed supervision for surface-defect detection: from weakly to fully supervised learning.* [arXiv:2104.06064](https://arxiv.org/abs/2104.06064)

[2] V. Zavrtanik, M. Kristan, D. Skočaj. *DSR – A dual subspace re-projection network for surface anomaly detection.* [arXiv:2208.01521](https://arxiv.org/abs/2208.01521)
