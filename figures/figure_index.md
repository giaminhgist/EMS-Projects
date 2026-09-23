# Figure Index — EMS-Projects Presentation (Learned Normative Gaze Modeling)

Mỗi mục: gợi ý slide, figure, câu hỏi khoa học, kết luận được dữ liệu hỗ trợ, caption tiếng Anh, speaker notes tiếng Việt, giới hạn diễn giải.

Lưu ý port: Figures 1–3 tái lập từ cùng dataset EMS + cùng cleaning rules — mọi bảng số
(tables T01/T02) khớp chính xác với bộ reference (EMS-Project/presentation, diff = 0).
Figures 4–5 được tính lại từ các run đã commit của repo này (EXP-PROP-001/003/004, seed 42);
số liệu trong index này lấy từ chính các tables đã sinh.

---

## Figure 1 — Dataset

### Slide: "The EMS dataset" — Figure_1
- **Câu hỏi**: Dataset gồm ai, xem gì, chia fold thế nào?
- **Kết luận**: 160 labelled (80 HC / 80 SZ) + 48 official-test không nhãn; 100 stimuli / 4 categories (22 social, 31 natural, 15 synthetic, 32 manipulated); 4 official folds không cân bằng hoàn toàn (Set_1 = 24 HC / 16 SZ).
- **Caption (EN)**: "EMS eye-tracking dataset (Song et al., IEEE TNNLS 2024). (a) 160 train/valid subjects (80 HC, 80 SZ) with labels; below (a), one example stimulus per category. (b) 100 free-viewing stimuli (5 s each) across four categories. (c) Official 4-fold split (40 subjects per fold); fold composition is not perfectly balanced (Set_1 has 24 HC / 16 SZ)."
- **Speaker notes (VI)**: "Nêu nhanh nguồn dữ liệu công khai EMS, kích thước nhỏ (n=160 có nhãn) — đây là lý do phải chống leakage cẩn thận. Fold không cân bằng nên Balanced Accuracy quan trọng hơn Accuracy."
- **Giới hạn**: Không có metadata tuổi/giới/clinical scores trong repo → không vẽ được; labels của official test bị giữ kín nên không được tự gán.

### Slide: "Gaze signatures: HC vs SZ" — Figure_2
- **Câu hỏi**: HC và SZ khác nhau ở các thống kê gaze cơ bản nào, và scanpath trên cùng stimulus trông thế nào?
- **Kết luận**: Ở mức subject (n=80/80): SZ ít fixation hơn (d≈+0.99), fixation dài hơn (d≈−1.02), pupil nhỏ hơn (d≈+0.39), dispersion hẹp hơn (d≈+0.69, không vẽ — số ở caption/bảng T01.03) — "restricted visual pattern"; scanpath trên cùng stimulus ngắn và co cụm hơn, hiệu ứng phụ thuộc stimulus.
- **Caption (EN)**: "Top (a–c): subject-level gaze statistics on cleaned fixations (unit = subject, n = 80 per group; Welch's t and Cohen's d computed on subject aggregates, not individual fixations, Cohen's d = HC − SZ). (a) Fixations per subject: HC 1,463 vs SZ 1,234, Welch p = 4e−09, d = +0.98. (b) Mean fixation duration: HC 271 vs SZ 326 ms, p = 2e−09, d = −1.02. (c) Mean pupil size: HC 1,356 vs SZ 1,124 a.u., p = 0.015, d = +0.38. Mean dispersion (distance to centroid, not plotted): HC 200 vs SZ 179 px, p = 2.5e−05, d = +0.68. (d) Scanpaths of representative HC (top row) and SZ (bottom row) subjects on the same four stimuli (one per category). Stimulus selection: per category, the stimulus whose mean fixation count across train subjects is closest to the category median. Subject selection: the subject of each group whose fixation count on that stimulus is closest to the group median. Square = first fixation; lines connect fixations in viewing order."
- **Speaker notes (VI)**: "Đây là động lực thiết kế feature: SZ quét ít hơn, chậm hơn, hẹp hơn — và abnormality phụ thuộc stimulus → cần normative conditioning theo stimulus. Nhấn mạnh: mọi suy diễn ở mức subject để không pseudo-replicate; quy tắc chọn subject/stimulus minh bạch (median rule)."
- **Giới hạn**: Không hiệu chỉnh confound (thuốc, tuổi) — EDA; 2 subject per stimulus trong scanpath chỉ là minh họa.

---

## Figure 3 — Hand-crafted features

### Slide: "Which features discriminate?" — Figure_3
- **Kết luận**: spa_entropy, geo_scanpath_len, tem_fix_rate, spa_fix_count có |d| ≈ 1.0–1.07 (subject level); nhóm pupil và geo chiếm đa số top features.
- **Caption (EN)**: "(a) Cohen's d (HC − SZ) for all 45 features at subject level (n=80/80); stars = Welch p<0.05/0.01/0.001. (b) Rainclouds for one representative feature per group. (c) Per-subject category means of four features (mean ± SEM over subjects). The HC–SZ gap varies by category (e.g., tem_dur_mean differs most in social and manipulated images), motivating stimulus-conditioned modeling."
- **Speaker notes (VI)**: "Bức tranh nhất quán: SZ quét ít hơn, chậm hơn, hẹp hơn — và cùng một feature, giá trị phụ thuộc stimulus, nên normalize theo stimulus chứ không chỉ toàn cục. Đây là cơ sở cho giả thuyết normative."
- **Giới hạn**: Effect size đơn biến; không suy diễn nhân quả.

---

## Figure 4 — Latent distribution

### Slide: "Learned latent space vs the HC normative bank" — Figure_4
- **Kết luận**: (a–b) PCA (fit trên training-HC reference, PC1 giải thích 17.0% variance của reference): eval-HC nằm trong vùng mật độ HC reference, eval-SZ lệch rõ dọc PC1 (Welch p = 2.4e−13, d = −1.27) và PC2 (p = 9e−08, d = +0.89). (d) Heatmap subject × stimulus: SZ cao hơn HC một cách hệ thống (subject-mean RMS 1.12 vs 1.03), deviation cao rải rác theo subject, không tập trung ở một category. (c) Effective rank của train-HC covariance = 16.6 — không collapse. (e) Train-HC mean ‖z−μ‖ = 3.38 ± 0.19 — tập trung quanh bank nhưng không suy biến về 0.
- **Caption (EN)**: "(a) Joint PCA of per-stimulus bank-centered encodings (z − μ_s). PCA is fit on the training-fold HC reference (re-encoded with the final encoder; grey density), then evaluation subjects (out-of-fold, seed 42) are projected. Points/ellipses: subject means over valid stimuli. PC1 explains 17.0% of reference variance; axes are not hand-crafted features. (b) Subject-mean PC1 distributions. (c) Effective rank (participation ratio) of the train-HC latent covariance — well above 1, i.e., no collapse (this suite has no λ_norm; see missing_artifacts.md). (d) Subject × stimulus RMS standardized residual sqrt(mean_k ((z_k−μ_k)/σ_k)²) heatmap for 60 evaluation subjects (30 HC with the 15 lowest/highest subject-mean deviation each, 30 SZ likewise), stimuli sorted by category; missing pairs left blank. (e) Train-HC per-subject mean ‖z−μ‖: concentrated around the bank but not zero."
- **Speaker notes (VI)**: "Không gộp encoder của các fold/seed khác nhau vào chung một PCA — mỗi model một không gian; đây là seed 42. Ký hiệu: z trước comparator, d sau comparator, h sau pooling. Suite này đã bỏ λ_norm nên panel (c,e) là chẩn đoán single-model: rank hiệu dụng 16.6 chứng tỏ không collapse, khoảng cách train-HC tập trung (3.38 ± 0.19) chứng tỏ bank có ý nghĩa."
- **Giới hạn**: PCA 2D chỉ giữ ~27.2% variance của reference (17.0 + 10.2); không suy distance 2D thành distance không gian gốc. Heatmap là mô tả (60/160 subject được chọn theo quy tắc min/max deviation).

---

## Figure 5 — Importance & XAI

### Slide: "What drives the prediction?" — Figure_5
- **Kết luận**: Permutation (hoán đổi cả trajectory feature giữa subjects, giữ stimulus index; bank đóng băng; fold Set_1 val): cả learned (mlp_deepset) lẫn hard (z_mean) đều dựa nhiều vào pupil (ΔAUC 0.266 / 0.166), scanpath geometry (0.088 / 0.141) và spatial center (0.062 / 0.048); top-15 feature gồm toàn bộ 6 feature pupil trước tiên, rồi spa_entropy, spa_center_frac, geo_scanpath_len. Attention (mlp_attn) khá đều giữa các stimuli (0.0084–0.0138) và giữa categories/nhóm; leave-one-stimulus-out gần như không đổi AUC (drop trung bình 0.0037, max 0.0126; Pearson r = 0.34 giữa attention và AUC drop) — không có stimulus "nòng cốt", model dựa vào toàn bộ tập stimulus.
- **Caption (EN)**: "(a) Feature-family permutation importance of P(SZ) through the full pipeline (mlp_deepset vs z_mean, fold Set_1 validation, n=40; 8 repeats; the whole per-stimulus trajectory of a feature is swapped between subjects so stimulus structure is preserved; bank and weights frozen). (b) Top-15 individual features, mlp_deepset (10 repeats ± SD). ΔAUC = baseline AUC − permuted AUC. (c) Leave-one-stimulus-out AUC drop vs mean attention (mlp_attn Set_0 model, n=40 val; Pearson r). (d) Top-30 stimuli by mean attention weight (mlp_attn, pooled out-of-fold), colored by category. (e) Attention by category × group (mlp_attn)."
- **Speaker notes (VI)**: "Importance đo qua toàn bộ pipeline (encoder+comparator+pooling+head) — không đặt tên một latent dimension là một hand-crafted feature. Attention đơn thuần chưa đủ để kết luận importance; panel c cho thấy tín hiệu nằm ở tổng hợp toàn bộ stimuli. Lưu ý panel c–e dùng mlp_attn vì chỉ config này có attention pooling; panel a–b là model chính mlp_deepset."
- **Giới hạn**: Chỉ tính trên valid exposures; same stimulus set khi so sánh; permutation trên Set_1 val (n=40) nên std lớn ở một số feature.

---

## Bố cục slide gợi ý (5 slides chính)

1. EMS dataset (Figure_1)
2. HC vs SZ gaze signatures: distributions + scanpaths (Figure_2)
3. Feature discriminability (Figure_3)
4. Latent space: PCA + heatmap + rank/distance diagnostics (Figure_4)
5. Importance (Figure_5)
