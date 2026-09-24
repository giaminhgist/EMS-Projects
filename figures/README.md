# Presentation figure suite — EMS-Projects

Bộ figures phục vụ presentation với supervisor về *Learned Stimulus-Conditioned
Normative Modeling* trên dataset EMS. Mọi số liệu được **tính lại từ run
artifacts đã commit** (checkpoints, predictions.csv, summaries) của repo này;
không có số liệu nào bịa.

Bộ này được port từ reference suite (EMS-Project/presentation) sang repo
EMS-Projects: Figures 1–3 dùng cùng dataset + cùng cleaning rules nên các
tables T01/T02 khớp chính xác với reference (đã đối chiếu, diff = 0);
Figures 4–5 dùng chính các model của repo này (EXP-PROP-001 mlp_deepset,
EXP-PROP-003 mlp_attn, EXP-PROP-004 z_mean, seed 42).

## Nội dung

```
figures/
├── figure/                  # 5 figures — PNG 300 dpi + SVG, đặt tên Figure_1..Figure_5
│   ├── Figure_1.png/.svg    # dataset overview (subjects, stimuli, official folds)
│   ├── Figure_2.png/.svg    # gaze signatures: HC vs SZ distributions + scanpaths
│   ├── Figure_3.png/.svg    # effect sizes của 45 features + rainclouds + category profiles
│   ├── Figure_4.png/.svg    # normative latent: PCA + category deviation + heatmap + distance diagnostic
│   └── Figure_5.png/.svg    # feature + stimulus importance (Nature style)
├── scripts/                 # code tái lập (entry point: run_all.py)
├── tables/                  # CSV/JSON nguồn cho từng figure
├── cache/                   # intermediate tensors (fixations + latent exports)
├── figure_manifest.csv      # figure → nguồn, run/config, protocol, seed/fold, script
├── figure_index.md          # slide gợi ý, captions (EN), speaker notes (VI), giới hạn
└── missing_artifacts.md     # phần thiếu + lệnh bổ sung
```

## Chạy

```bash
cd /root/EMS-Projects

# toàn bộ (lần đầu build cache fixations ~3 phút, latent exports ~1 phút)
.venv/bin/python figures/scripts/run_all.py

# một nhóm
.venv/bin/python figures/scripts/run_all.py --groups 05

# chạy trực tiếp một script
.venv/bin/python figures/scripts/make_latent_figs.py

# regenerate manifest + gallery (contact sheet) sau khi đổi figures
.venv/bin/python figures/scripts/make_manifest_gallery.py
```

Cache được bỏ qua nếu đã tồn tại; `--force-cache` để rebuild fixations.

## Dependencies

Đúng môi trường của repo (`.venv`, `uv sync`): python ≥3.12, numpy, pandas,
scipy, scikit-learn, matplotlib, torch (CPU đủ — model ~201k params),
openpyxl (đọc xlsx fixation files).

## Quy ước chống sai số (đọc trước khi sửa figure)

- **Đơn vị thống kê**: suy diễn HC/SZ luôn ở mức *subject* (n=80/80), không
  dùng hàng nghìn fixation làm quan sát độc lập.
- **P1**: mean qua seeds của mean 4 folds; fold-mean ≠ pooled AUC (ROC dùng
  pooled, bảng số dùng fold-mean — đều ghi nhãn rõ).
- **Checkpoint**: mọi inference dùng `best.pt` nguyên trạng, giữ nguyên bank
  buffers (không refresh), không retrain.
- **Verification**: script latent (Figure_4) dùng nguyên trạng các exports
  bank/z từ checkpoint `best.pt` trong `cache/latent`; probs của export khớp
  `predictions.csv` đã commit đến 1e−16 (đã kiểm tra).
- **XAI**: ranking tính trên fold Set_1, đánh giá trên fold Set_0 (partition
  riêng); permutation hoán đổi cả feature trajectory giữa subjects.
- **Không impute giả**: NaN → 0 trong standardized space có nghĩa "= HC norm",
  mask loại pair thiếu; không trình bày như quan sát sinh học thật.

## Nguồn dữ liệu chính

- Run artifacts: `outputs/experiments/{EXP-PROP-001,003,004}/full45/seed42/
  foldSet_*/` (best.pt, config.json, predictions.csv, metrics.jsonl).
- Dữ liệu: `original_dataset/EMS/`, `processed_dataset/` (features, metadata,
  quality_report).
- Docs đối chiếu: `docs/features45.md`, `docs/experiments/EXP-PROP-*.md`,
  `docs/experiments/baselines.md`.
