# Paper text for the figure suite — EMS-Projects

File này gắn mỗi figure trong `figures/figure/` với **văn bản sẵn sàng đưa
vào paper** (caption + prose cho phần Results), viết bằng tiếng Anh. Phần
mô tả và ghi chú phương pháp viết bằng tiếng Việt.

**Nguồn số liệu**: mọi con số trong các đoạn prose dưới đây lấy nguyên văn
từ `figures/tables/T*.csv` và `EXPERIMENTS_RESULTS.md` (đã commit); không
có số nào được ngoại suy hay làm tròn theo hướng có lợi. Nếu prose mâu
thuẫn với table, prose sai — sửa prose, không sửa số.

## Bản đồ figure → section của paper

| Figure | Nội dung | Section gợi ý trong paper |
|---|---|---|
| Figure_1 | EMS dataset: subjects, stimuli, official folds | Methods — Dataset (hoặc Results — Dataset overview) |
| Figure_2 | Gaze signatures HC vs SZ: distributions + scanpaths | Results — Group differences in gaze signatures |
| Figure_3 | Effect sizes 45 features + rainclouds + category profiles | Results — Which features discriminate (motivation cho stimulus-conditioned modeling) |
| Figure_4 | Latent space vs HC normative bank (PCA, deviation, heatmap, diagnostics) | Results — Learned deviation from the HC normative bank |
| Figure_5 | Feature + stimulus importance (permutation, attention, LOO) | Results — What drives the prediction |

Bảng kết quả chính (AUC 0.9403 ± 0.0156 của EXP-PROP-001, các ablation,
baselines, so sánh với MSNet đã công bố) là **Table**, không phải figure —
xem `EXPERIMENTS_RESULTS.md`; các prose dưới đây cross-reference bảng đó.

## Quy ước viết (áp dụng cho mọi prose bên dưới)

1. **Đơn vị suy diễn = subject** (n = 80 HC / 80 SZ). Không bao giờ viết
   câu gợi ý fixation là quan sát độc lập.
2. **Dấu của Cohen's d**: `d = HC − SZ` (d dương = HC lớn hơn). Khi viết
   "larger/longer" phải khớp dấu.
3. **Figures 4–6**: mọi inference dùng checkpoint `best.pt` nguyên trạng
   (bank buffers giữ nguyên, không refresh, không retrain), protocol P1
   out-of-fold, seed 42; đánh giá trên fold validation tương ứng. Không
   gộp encoder của các fold/seed khác vào chung một PCA.
4. **Interpretation**: association, không causal; deviation so với HC norm
   ≠ chẩn đoán lâm sàng. Giữ câu này (hoặc tương đương) trong paper.
5. **So sánh MSNet**: số đã công bố (AUC 0.8972), không retrain, unpaired —
   prose không được viết thành "significantly better".

---

## Figure 1 — Dataset overview

### Mô tả

- **Nội dung**: (a) 160 subjects có nhãn (80 HC / 80 SZ) + ví dụ một
  stimulus mỗi category; (b) 100 stimuli free-viewing (5 s mỗi stimulus)
  phân bố theo 4 category; (c) composition của 4 official folds.
- **Nguồn**: `tables/T01.01_fold_composition.csv`, `T01.01_stimulus_counts.csv`;
  script `scripts/make_dataset_figs.py`. Không cần model.
- **Điểm cần nhấn**: fold không cân bằng hoàn toàn (Set_1 = 24 HC / 16 SZ)
  → lý do báo cáo Balanced Accuracy / SEN / SPEC kèm AUC; 48 official-test
  subjects không nhãn không được tự gán label.

### Paper caption (draft)

> **Figure 1.** Overview of the EMS eye-tracking dataset (Song et al., IEEE
> TNNLS 2024). (a) The 160 training/validation subjects with labels
> (80 healthy controls, HC; 80 schizophrenia, SZ) and one example stimulus
> per category. (b) The 100 free-viewing stimuli (5 s each) across four
> categories: social (22), natural (31), synthetic (15), manipulated (32).
> (c) Composition of the official four-fold split (40 subjects per fold);
> folds are not perfectly balanced (Set_1 contains 24 HC / 16 SZ), so we
> report sensitivity, specificity, and balanced accuracy alongside AUC.

### Results prose (draft — Methods/Dataset)

> We use the publicly available EMS eye-tracking dataset [Song et al.,
> TNNLS 2024], which records free viewing of 100 natural images (5 s each)
> in 160 labelled subjects (80 HC, 80 SZ) plus 48 unlabelled official-test
> subjects. The 100 stimuli fall into four categories — social (22),
> natural (31), synthetic (15), and manipulated (32) — and every subject
> viewed all 100 stimuli, yielding one per-(subject, stimulus) gaze
> recording. The dataset provides an official subject-level four-fold
> split; the folds are not perfectly balanced by diagnosis (e.g., Set_1
> has 24 HC vs 16 SZ, Figure 1c), so we evaluate with ROC-AUC together
> with accuracy, sensitivity, and specificity at a fixed 0.5 threshold
> (SZ = positive). The official-test labels are withheld; we use the
> official-test split only as specified by the benchmark and never
> evaluate on it.

### Ghi chú cho paper

- Giới hạn cần giữ trong text: không có metadata tuổi/giới/clinical
  scores trong repo → không kiểm soát được confound nhân khẩu; kết luận
  là association.
- Caption phải giữ câu "folds are not perfectly balanced" — nó biện minh
  cho metric báo cáo.

---

## Figure 2 — Gaze signatures: HC vs SZ

### Mô tả

- **Nội dung**: (a–c) phân phối subject-level của 3 thống kê gaze trên
  fixations đã clean (n = 80/nhóm): fixations/subject, mean fixation
  duration, mean pupil size; (d) scanpath của 2 subject đại diện (1 HC +
  1 SZ) trên cùng 4 stimulus (1 mỗi category, chọn bằng median rule —
  `tables/T01.04_selected_stimuli.csv`).
- **Nguồn**: `tables/T01.03_subject_stats.csv`, cache `fixations_cleaned.pkl`;
  script `scripts/make_dataset_figs.py`. Không dùng model.
- **Số chính**: fixations 1,463 vs 1,234 (Welch p = 4e−09, d = +0.98);
  duration 271 vs 326 ms (p = 2e−09, d = −1.02); pupil 1,356 vs 1,124 a.u.
  (p = 0.015, d = +0.38); dispersion 200 vs 179 px (p = 2.5e−05, d = +0.68,
  không vẽ, nằm trong caption/table).

### Paper caption (draft)

> **Figure 2.** Subject-level gaze statistics on cleaned fixations (unit
> of analysis = subject, n = 80 per group; Welch's t and Cohen's d on
> subject aggregates, d = HC − SZ). (a) Fixations per subject:
> HC 1,463 vs SZ 1,234, p = 4e−09, d = +0.98. (b) Mean fixation duration:
> HC 271 vs SZ 326 ms, p = 2e−09, d = −1.02. (c) Mean pupil size:
> HC 1,356 vs SZ 1,124 a.u., p = 0.015, d = +0.38. (Mean dispersion,
> distance to fixation centroid, not plotted: HC 200 vs SZ 179 px,
> p = 2.5e−05, d = +0.68.) (d) Scanpaths of one representative HC (top)
> and one representative SZ (bottom) subject on the same four stimuli,
> one per category (stimulus = per category the one with mean fixation
> count closest to the category median across training subjects;
> subject = per group the one with fixation count on that stimulus
> closest to the group median). Square = first fixation; lines connect
> fixations in viewing order.

### Results prose (draft — "Group differences in gaze signatures")

> We first characterize group differences at the subject level
> (n = 80 per group), aggregating each subject's cleaned fixations across
> the 100 stimuli before any test (Figure 2). Relative to HC, SZ subjects
> produce fewer fixations (1,234 vs 1,463; Welch p = 4e−09, d = +0.98),
> longer fixations (326 vs 271 ms; p = 2e−09, d = −1.02), smaller pupils
> (1,124 vs 1,356 a.u.; p = 0.015, d = +0.38), and narrower gaze
> dispersion (179 vs 200 px mean distance to centroid; p = 2.5e−05,
> d = +0.68) — a "restricted scanning" pattern consistent with the
> eye-tracking literature in schizophrenia. Representative scanpaths on
> the same four stimuli (Figure 2d) show the same pattern at a glance:
> SZ scanpaths are shorter and more clustered, and the size of the
> HC–SZ gap visibly depends on the stimulus being viewed. This
> stimulus dependence motivates modelling abnormality as a deviation
> conditioned on the stimulus rather than as a global summary statistic.

### Ghi chú cho paper

- Giữ câu "unit of analysis = subject" — chống pseudo-replication.
- Scanpath panel là minh họa (2 subjects); không dùng để suy diễn.
- Không viết "SZ nhìn ít hơn" thành claim causal; thuốc/độ tuổi không
  được kiểm soát (nêu ở Limitations).

---

## Figure 3 — Effect sizes of the 45 features

### Mô tả

- **Nội dung**: (a) Cohen's d (HC − SZ) của cả 45 hand-crafted features ở
  mức subject (n = 80/80), sắp theo |d|, tô màu theo family, sao = Welch
  p < 0.05/0.01/0.001; (b) raincloud của một feature đại diện mỗi family;
  (c) subject-mean theo category của 4 feature (tem_dur_mean,
  spa_dispersion, geo_scanpath_len, pup_mean; mean ± SEM).
- **Nguồn**: `tables/T02.02_effect_sizes.csv`, `T02.03_category_profiles.csv`;
  script `scripts/make_feature_figs.py`.
- **Số chính**: top |d|: spa_entropy 1.07, geo_scanpath_len 1.05,
  tem_fix_rate 1.04, spa_fix_count 1.02, tem_trans_entropy 1.01 (mọi
  p < 1e−8); nhóm geo/tem/spatial chiếm đa số top features; dấu d = HC − SZ
  (tem_dur_mean d = −0.89: SZ fixation dài hơn).

### Paper caption (draft)

> **Figure 3.** Discriminability of the 45 hand-crafted gaze features at
> subject level (n = 80 per group; Cohen's d = HC − SZ; stars = Welch
> p < 0.05/0.01/0.001). (a) Effect sizes for all 45 features, colored by
> feature family. (b) Rainclouds for one representative feature per
> family. (c) Per-subject category means of four features (mean ± SEM
> over subjects). The HC–SZ gap varies by category (e.g., tem_dur_mean
> differs most in social and manipulated images), motivating
> stimulus-conditioned modelling.

### Results prose (draft — "Which features discriminate")

> To ground the modelling choices, we computed Cohen's d at subject level
> for all 45 hand-crafted gaze features (Figure 3a). The largest effects
> cluster in scanpath geometry and temporal structure — spa_entropy
> (d = 1.07), geo_scanpath_len (d = 1.05), tem_fix_rate (d = 1.04),
> spa_fix_count (d = 1.02) — with pupil and spatial-center features also
> discriminating. The sign pattern is coherent: SZ scan fewer, slower
> (tem_dur_mean d = −0.89), and within a narrower region. Importantly,
> the same feature shows different HC–SZ gaps across stimulus categories
> (Figure 3c; e.g., tem_dur_mean separates most in social and manipulated
> images). A single global summary of a subject's gaze therefore
> conflates stimulus content with group signal, which is precisely the
> motivation for conditioning the normative model on the stimulus.

### Ghi chú cho paper

- Dấu của d phải nhất quán với Figure 2 (duration: SZ dài hơn → d âm).
- Đây là phân tích đơn biến; viết rõ không chọn feature bằng test data —
  mọi thống kê tính trên train subjects của official split (không dùng
  test labels).

---

## Figure 4 — Learned latent space vs the HC normative bank

### Mô tả

- **Nội dung**: (a) PCA của per-stimulus bank-centered encodings
  (z − μ_s), PCA fit trên training-fold HC reference, các subject đánh
  giá (out-of-fold) được project vào; điểm/ellipse = subject mean;
  (b) phân phối subject-mean PC1; (c) subject-mean RMS standardized
  residual theo category (violin, n = 80/nhóm); (d) heatmap subject ×
  stimulus của RMS standardized residual (60 subjects chọn theo quy tắc
  min/max deviation); (e) train-HC per-subject mean ‖z − μ‖.
- **Nguồn**: `cache/latent/mlp_deepset__seed42__*.npz` (z, bank, ref_z),
  `tables/T05.01_*.csv`, `T05.01_deviation_matrix.npy`; script
  `scripts/make_latent_figs.py`. Model: EXP-PROP-001 (mlp_deepset),
  seed 42, protocol P1 out-of-fold.
- **Số chính**: PC1 17.0% variance của reference (PC2 10.2%); eval-SZ
  lệch khỏi eval-HC dọc PC1 (Welch p = 2.4e−13, d = −1.27) và PC2
  (p = 9e−08, d = +0.89); deviation theo category SZ > HC ở cả 4
  category (1.09–1.14 vs 1.02–1.03, Welch p < 1e−6 mỗi category,
  `T05.01_category_deviation.csv`); subject-mean RMS 1.1199 vs 1.0262
  (`T05.01_norm_diagnostics.csv`); train-HC ‖z − μ‖ = 3.38 ± 0.19;
  effective rank (participation ratio) của train-HC covariance = 16.6
  (latent dim 128) — không collapse.

### Paper caption (draft)

> **Figure 4.** The learned latent space relative to the HC normative
> bank (EXP-PROP-001 mlp_deepset, seed 42, out-of-fold evaluation).
> (a) Joint PCA of per-stimulus bank-centered encodings (z − μ_s). The
> PCA is fit on the training-fold HC reference (grey density, re-encoded
> with the final encoder); evaluation subjects are projected. Points and
> ellipses are subject means over valid stimuli; PC1 explains 17.0% of
> the reference variance (PC2: 10.2%); axes are latent, not hand-crafted
> features. (b) Subject-mean PC1 distributions (Welch p = 2.4e−13,
> d = −1.27; PC2: p = 9e−08, d = +0.89). (c) Subject-mean RMS
> standardized residual sqrt(mean_k ((z_k − μ_k)/σ_k)²) by stimulus
> category (violins over subjects, n = 80 per group): SZ is elevated in
> all four categories (Welch p < 1e−6 each; Table T05.01). (d) Subject ×
> stimulus RMS standardized residual for 60 evaluation subjects (30 HC
> with the lowest/highest subject-mean deviation, 30 SZ likewise),
> stimuli sorted by category; missing pairs left blank. (e) Train-HC
> per-subject mean ‖z − μ‖: concentrated around the bank but not
> degenerate (3.38 ± 0.19; participation ratio of the train-HC
> covariance = 16.6, i.e., no collapse).

### Results prose (draft — "Learned deviation from the HC normative bank")

> We next ask where SZ subjects live in the latent space learned by the
> main model relative to the HC normative bank (Figure 4). Because each
> checkpoint defines its own latent geometry, all projections use a
> single model (seed 42) and out-of-fold subjects; the PCA is fit on the
> training-fold HC reference only, so the displayed structure is a
> property of the HC norm, not of the test labels. Evaluation HC
> subjects fall inside the HC reference density, while evaluation SZ
> subjects are systematically displaced along PC1 (Welch p = 2.4e−13,
> d = −1.27) and PC2 (p = 9e−08, d = +0.89). The standardized deviation
> is elevated in SZ across all four stimulus categories (subject-mean
> RMS 1.09–1.14 vs 1.02–1.03, p < 1e−6 per category; Figure 4c), i.e.,
> the abnormality is not confined to one stimulus type, and the
> subject × stimulus heatmap (Figure 4d) shows SZ elevations that are
> systematic at the subject level and scattered across stimuli rather
> than tied to one category. Two diagnostics support that the bank is a
> meaningful norm: train-HC encodings concentrate around their
> per-stimulus means (mean ‖z − μ‖ = 3.38 ± 0.19; Figure 4e) without
> collapsing to a point (participation ratio 16.6 of the 128-d train-HC
> covariance), and evaluation HC remain close to the bank (subject-mean
> RMS 1.03), as expected for healthy reference members.

### Ghi chú cho paper

- PCA 2D chỉ giữ ~27.2% variance của reference — không suy khoảng cách
  2D thành khoảng cách không gian gốc; câu này nên vào Limitations.
- Heatmap (d) là mô tả: 60/160 subjects chọn theo quy tắc min/max
  deviation — ghi rõ trong caption (đã có).
- Đừng gọi PC1 là "schizophrenia axis" — nó là trục variance của HC norm.
- Deviation ≠ chẩn đoán: giữ câu "a deviation from the HC norm is not a
  diagnosis" ở Discussion.

---

## Figure 5 — Feature and stimulus importance

### Mô tả

- **Nội dung**: (a) permutation importance theo feature family qua toàn bộ
  pipeline (mlp_deepset vs z_mean, fold Set_1 val, n = 40, 8 repeats);
  (b) top-15 feature riêng lẻ của mlp_deepset (10 repeats, error bar ± SEM);
  (c) leave-one-stimulus-out AUC drop vs mean attention (mlp_attn,
  Set_0 val, Pearson r); (d) top-30 stimuli theo mean attention (pooled
  out-of-fold), tô màu theo category; (e) attention theo category × group.
- **Nguồn**: checkpoints EXP-PROP-001 + EXP-PROP-004 (permutation trên
  Set_1 val); exports mlp_attn EXP-PROP-003; `tables/T06.01*.csv`; script
  `scripts/make_xai_figs.py`.
- **Số chính**: ΔAUC = AUC baseline − AUC permuted; pupil 0.266 (learned)
  / 0.166 (z_mean), geo 0.088 / 0.141, spa_center 0.062 / 0.048
  (`T06.01b_importance_families.csv`); attention gần uniform giữa stimuli
  (0.0084–0.0138), LOO drop trung bình 0.0037 (max 0.0126), Pearson
  r = 0.34 giữa attention và AUC drop — không có stimulus "nòng cốt".
- **Protocol XAI**: hoán đổi cả feature trajectory giữa subjects (giữ
  stimulus index), bank đóng băng; ranking trên Set_1, đánh giá trên
  Set_0 (partition riêng).

### Paper caption (draft)

> **Figure 5.** What drives the prediction. (a) Feature-family
> permutation importance of P(SZ) through the full pipeline
> (mlp_deepset vs the hard z_mean deviation, fold Set_1 validation,
> n = 40; 8 repeats; the whole per-stimulus trajectory of a feature is
> swapped between subjects so stimulus structure is preserved; bank and
> weights frozen). ΔAUC = baseline AUC − permuted AUC. (b) Top-15
> individual features, mlp_deepset (10 repeats, error bars ± SEM). (c)
> Leave-one-stimulus-out AUC drop vs mean attention weight (mlp_attn,
> Set_0 model, n = 40 validation subjects; Pearson r = 0.34).
> (d) Top-30 stimuli by mean attention (mlp_attn, pooled out-of-fold),
> colored by category. (e) Attention by category × group (mlp_attn).

### Results prose (draft — "What drives the prediction")

> To attribute the model's decision, we permute whole feature
> trajectories across subjects while keeping the stimulus structure
> intact, measure the resulting AUC drop through the complete pipeline
> (encoder → bank comparison → pooling → head, bank and weights frozen;
> Figure 5a). Both the learned model and the hard z-deviation model rely
> most on the pupil family (ΔAUC 0.266 vs 0.166), followed by scanpath
> geometry (0.088 vs 0.141) and spatial center (0.062 vs 0.048);
> positional statistics contribute essentially nothing (spa_pos −0.013 /
> −0.039). At the individual-feature level (Figure 5b), the six pupil
> features dominate the top-15, followed by spa_entropy,
> spa_center_frac, and geo_scanpath_len — consistent with the
> subject-level effect sizes of Figure 3. Attention weights
> (mlp_attn) are near-uniform across stimuli (0.0084–0.0138) and across
> categories and groups (Figure 5d–e), and leaving one stimulus out
> changes the AUC by 0.0037 on average (max 0.0126; Figure 5c), with
> only a weak attention–drop correlation (r = 0.34). The model therefore
> has no single "core" stimulus: the discriminative signal is spread
> over the whole stimulus set, and attention alone would understate
> this spread.

### Ghi chú cho paper

- Lưu ý trong text: panels a–b dùng mlp_deepset (model chính); c–e dùng
  mlp_attn vì chỉ config này có attention pooling — đừng đọc chung như
  một model.
- Permutation chỉ trên valid exposures (n = 40) → SD lớn ở vài feature;
  đây là phân tích giải thích, không phải test chọn feature.
- r = 0.34 là tương quan mô tả; không viết "attention predicts importance".

---

## Liên kết với các bảng kết quả chính (cross-references)

- **Main results table** (`EXPERIMENTS_RESULTS.md` §1): EXP-PROP-001
  (mlp_deepset) AUC 0.9403 ± 0.0156, ACC 0.8494, SEN 0.8229, SPEC 0.8717;
  best baseline fnn 0.9176; so sánh MSNet đã công bố 0.8972 (unpaired).
  Figures 4–6 giải thích *tại sao* model này hoạt động — prose của chúng
  nên được đặt ngay sau bảng chính.
- **Ablations** (`EXPERIMENTS_RESULTS.md` §2): 002 mean-pool 0.9383,
  003 attn-pool 0.9299, 004 hard z 0.9073, 005 max-pool 0.9304 — ủng hộ
  phát biểu "learned deviation + deepset pooling là cấu hình mạnh nhất";
  Figure 5a so sánh trực tiếp 001 vs 004 ở mức feature.
- **Diễn giải tổng**: chuỗi lập luận Figures 2→3→4→5 = (i) SZ khác HC ở
  gaze thống kê và sự khác biệt phụ thuộc stimulus → (ii) feature nào
  mang tín hiệu → (iii) latent deviation so với HC norm tách được nhóm và
  bền theo category → (iv) model dựa vào pupil/geometry và tín hiệu trải
  đều khắp stimulus set (không có stimulus "nòng cốt").
