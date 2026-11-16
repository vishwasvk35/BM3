# BM3 vs BM3SimSiam: code-verified comparison

Scope: read-only. Nothing was modified. No PyTorch is installed in this environment, so nothing was executed. Findings come from reading the code, the configs, git history, and from parameter counts computed by hand and checked against the logged counts.

Paths are relative to `/home/vishwas/Desktop/RDS/BM3_implementation`. Line numbers refer to the files at HEAD (`94ff3e1`).

## 1. Summary

1. **BM3 has no momentum target network.** `src/models/bm3.py` contains no EMA or momentum update (`grep -i "momentum\|ema"` matches nothing in the code). Its target is the online network's own output, cloned under `torch.no_grad()` and passed through `F.dropout` (bm3.py:111-124). The loss then applies `.detach()` on that target (bm3.py:139-149). BM3 is already a stop-gradient plus dropout-view bootstrap with a predictor. The claims in `docs/BM3_SIMSIAM.md` ("momentum-updated target network", `θ_t = m*θ_t + (1-m)*θ_o`, "2× model parameters", "momentum coefficient" hyperparameter) are wrong. The README's "dual-network architecture with momentum-based bootstrapping" (README.md:51) is also wrong.
2. **The real change is a replacement of the loss head and item representation, not of a target network.** BM3SimSiam does these things:
   - It removes the dropout-perturbed target views.
   - It removes the symmetric user-item loss with detached targets.
   - It removes the four multimodal alignment and intra-modality losses (`cl_weight`).
   - It removes the shared `nn.Linear(64,64)` predictor and the `+ h` ID residual.
   - It adds a fusion module, a 3-layer BatchNorm encoder, a 2-layer BatchNorm predictor, a separate linear user predictor, and a single-view SimSiam loss on item embeddings.
   - It changes the scoring path, so text and image features now enter the item representation at inference time. In BM3 they only shape the embeddings through the training loss.
3. **Unchanged:** the LightGCN propagation (copied verbatim), the user and ID embeddings, the pretrained text and image embeddings (trainable), the 384→64 and 4096→64 linear projections (moved into `MultiModalFusion`, same init), the adjacency construction, `EmbLoss`, the trainer, and the dot-product ranking at inference.
4. **The SimSiam loss as implemented is single-view.** `p` and `z` come from the same input in the same pass, with no augmentation and no second view (bm3_simsiam.py:268-278). The user-item loss `loss_rec` has no stop-gradient and no negatives (bm3_simsiam.py:301-311). Collapse is therefore possible. This is a likely contributor to the large accuracy gap reported in `COMPARISON_BM3_vs_BM3SimSiam.md` (Recall@20 0.0765 vs 0.0869 after 600 epochs, and 0.037 vs 0.084 at 75 epochs). It is a hypothesis; no ablation was run.
5. **Parameters are almost identical.** BM3SimSiam has +169,280 parameters on both Baby and Sports (+0.50% and +0.20%). These counts match the logs exactly (section 5).

## 2. Evidence

### 2.1 BM3 (`src/models/bm3.py`)

| Item | Detail | Lines |
|---|---|---|
| Hyperparameters read | `embedding_size`, `n_layers`, `reg_weight`, `cl_weight`, `dropout`. `feat_embed_dim` is taken from `embedding_size`, so the YAML `feat_embed_dim` is ignored. | 32-37 |
| Embeddings | `user_embedding` (n_users×64), `item_id_embedding` (n_items×64), Xavier uniform | 44-47 |
| Predictor | One shared `nn.Linear(64,64)`, Xavier normal. It is applied to users, items, text and image. | 49, 52 |
| Modal parts | `image_embedding` / `text_embedding` from pretrained features (`freeze=False`). `image_trs` is Linear(4096→64) and `text_trs` is Linear(384→64), Xavier normal. | 54-61 |
| forward | LightGCN on ID embeddings only (n_layers, mean over layers). Returns `(u_g, i_g + h)`, where `h` is the raw item ID embedding (ID residual). | 91-100 |
| Online view | `u_online_ori, i_online_ori = forward()`. Text and image are `text_trs(text_embedding.weight)` and `image_trs(...)` over all items. | 104-109 |
| Target view | Under `torch.no_grad()`: clone, then `F.dropout(x, p=self.dropout)` for user, item, text and image. `u_target.detach()` on lines 113-114 discards its result and is a no-op. Gradient is blocked by `no_grad` and by `.detach()` inside the losses. `F.dropout` is called without `training=`, so it is always active. | 111-124 |
| Predictor use | Applied to `u_online_ori`, `i_online_ori`, `t_feat_online`, `v_feat_online`. | 126, 136, 142 |
| Loss terms | See the next table. | 139-152 |
| Inference | `predictor(u)`, `predictor(i)`, then `u[user] @ i.T` (dot product, 64-d). Text and image do not appear. | 154-159 |

BM3 loss terms (bm3.py:139-152). Every term is `1 - mean(cos(a, b))`.

| Name in code | Online side | Target side (detached) | Paper role | Weight |
|---|---|---|---|---|
| `loss_ui` | pred(user) | `i_target` (dropout) | L_rec | 1 |
| `loss_iu` | pred(item) | `u_target` (dropout) | L_rec | 1 |
| `loss_t` | pred(text proj) | `i_target` | L_align (inter-modality) | `cl_weight` = 2.0 |
| `loss_v` | pred(image proj) | `i_target` | L_align (inter-modality) | `cl_weight` = 2.0 |
| `loss_tv` | pred(text proj) | dropout(text proj) | L_ic (intra-modality). The name is misleading: it is text vs text. | `cl_weight` = 2.0 |
| `loss_vt` | pred(image proj) | dropout(image proj) | L_ic (intra-modality). The name is misleading: it is image vs image. | `cl_weight` = 2.0 |
| `reg_loss` | `EmbLoss(u_online_ori, i_online_ori)` over ALL users and items (propagated, includes `+h`) | none | L_reg | `reg_weight` = 0.1 |

`EmbLoss` (common/loss.py:38-52) is the sum of unsquared L2 (Frobenius) norms divided by the row count of the last argument. For BM3 that last argument is `i_online_ori`, so the divisor is `n_items`.

### 2.2 BM3SimSiam (`src/models/bm3_simsiam.py`, `src/models/simsiam_modules.py`)

| Item | Detail | Lines |
|---|---|---|
| Hyperparameters | `embedding_size`, `feat_embed_dim` (default 64), `n_layers`, `reg_weight`, `dropout` (read but never used afterwards), `simsiam_weight`, `simsiam_hidden_dim`, `simsiam_output_dim`. There is no `cl_weight`. | 45-53 |
| Embeddings | `user_embedding`, `item_id_embedding` (same as BM3). `text_embedding` and `image_embedding` are the same pretrained, trainable embeddings. | 63-85 |
| Fusion | `MultiModalFusion(384, 4096, feat_embed_dim=64, output_dim=None)`. It holds `text_proj` Linear(384→64) and `image_proj` Linear(4096→64) (Xavier normal), applies both, and concatenates them to 128-d. There is no final projection. | bm3_simsiam.py:87-93; simsiam_modules.py:~150-215 |
| Encoder | `SimSiamEncoder(192→256→256→128)` with BN and ReLU. Layers 1-2 have bias-free Linear plus BN (affine); layer 3 is Linear plus BN with `affine=False`. | bm3_simsiam.py:106-110; simsiam_modules.py:~14-90 |
| Predictor | `SimSiamPredictor(128→64→128)`: Linear (no bias), BN, ReLU, Linear (bias). | bm3_simsiam.py:113-117 |
| User predictor | `nn.Linear(64,128)`, Xavier normal. | 120-121 |
| forward | LightGCN on ID embeddings only, identical to BM3 (lines 224-241). Items become `cat([i_g, fused], -1)` (192-d). There is no `+ h` residual. Users stay 64-d. Fusion bypasses the graph. | 195-250 |
| Dead code | `item_features = cat([item_id_embedding.weight, fused])` (lines 217, 219) is never used. The docstring says "Add residual connection" at line 245, but the code concatenates. | 217-219, 245 |
| `calculate_simsiam_loss` | `z = encoder(x)`, `p = predictor(z)`, `loss = -mean(cos(p, z.detach()))`. | 268-278 |
| `calculate_recommendation_loss` | `user_pred = user_predictor(u)`, `item_pred = predictor(encoder(i))`, `loss = mean(1 - cos(user_pred, item_pred))`. There is no `detach` and no dropout. It is one-directional. | 296-311 |
| Reg | `EmbLoss(user_embedding.weight[users], item_id_embedding.weight[items])`. This uses raw, non-propagated embeddings for the batch only, so the divisor is the batch size. | 342-345 |
| Total | `loss_rec + simsiam_weight*loss_simsiam + reg_weight*loss_reg`. The weights are 1, 2.0 and 0.1. | 348-352 |
| Inference | `user_predictor(u)` against `predictor(encoder(item_embeddings))`, dot product, 128-d. Text and image enter the item side through the fused features. | 368-383 |

In `calculate_loss`, `batch_item_emb` is encoded twice per step, once in `loss_rec` and once in `loss_simsiam`. Both calls receive the same input, so this is wasted compute and a double update of the BN running statistics.

### 2.3 Differences from BM3's baseline config and global config

- `src/configs/model/BM3.yaml` was edited in commit `9e540d7`. The grid (`n_layers: [1,2]`, `dropout: [0.3,0.5]`, `reg_weight: [0.1,0.01]`) became single values (2, 0.3, 0.1), and `hyper_parameters` became `[]`. This makes the comparison like-for-like, but the baseline config is no longer the original BM3 grid.
- `src/configs/overall.yaml` was edited in the same commit: epochs 1000→75 and `stopping_step` 20→10. This affects both models.
- `src/utils/utils.py:36-44` was edited to map `BM3SimSiam` to the module `bm3_simsiam`. The trainer was not changed.
- `src/common/loss.py` gained an unused `SimSiamLoss` class (loss.py:65). `grep` finds no use in the models.

## 3. Side-by-side difference table

| Aspect | BM3 | BM3SimSiam |
|---|---|---|
| Target network | None separate. Same network output, cloned under `no_grad`, with dropout (bm3.py:111-124). **No momentum/EMA.** | None separate. SimSiam target is `z.detach()` (bm3_simsiam.py:275). No dropout. **No momentum/EMA.** |
| Source of the "second view" | Dropout noise on the embeddings (p=`dropout`, 0.3) | None. `p` and `z` both derive from the same input, so it is single-view. |
| Stop-gradient | Everywhere a target is used: `no_grad` plus `.detach()` (bm3.py:111, 139-149) | Only in the item self-loss (line 275). `loss_rec` has no stop-gradient. |
| Predictor | One shared Linear(64,64) for user, item, text, image (bm3.py:49) | Items: `predictor(encoder(·))`, a 3-layer BN MLP then a 2-layer BN MLP (192→256→256→128→64→128). Users: a separate Linear(64→128) (bm3_simsiam.py:106-121). |
| Item representation | `i_g + h` (64-d). `h` is the ID embedding residual. | `cat(i_g, fused_text_image)` (192-d). No `+ h`. |
| Text / image path | `text_trs`, `image_trs` produce 64-d features used only inside loss terms. They do not enter the item representation. | The same projections (in `MultiModalFusion`) are concatenated to 128-d and feed the item representation, so they affect scoring. |
| Graph propagation | LightGCN on ID embeddings (n_layers=2, mean pooling) | Same code. Fused features are NOT propagated through the graph, contrary to the docs diagram. |
| Rec loss | `loss_ui + loss_iu`: symmetric, detached dropout targets | `1 - cos(user_pred, item_pred)`: one direction, both sides receive gradient |
| Modal alignment loss (L_align) | `loss_t + loss_v` against `i_target` | **Removed** |
| Intra-modal loss (L_ic) | `loss_tv + loss_vt` against dropout view | **Removed.** Replaced by a single item-only `-cos(p, z.detach())`. |
| Regularization | `EmbLoss` on full propagated `u_g`, `i_g + h`. Divisor is n_items. | `EmbLoss` on batch raw ID embeddings. Divisor is the batch size. |
| Loss weights | rec 1, reg 0.1, cl 2.0 on the four modal terms | rec 1, simsiam 2.0, reg 0.1 |
| Dropout (0.3 in both configs) | Used to build the target views | Read into `self.dropout` (line 48) and never used |
| BatchNorm | None | In encoder (3 BN) and predictor (1 BN). Train and eval modes differ, so inference uses BN running averages. |
| Inference scoring | `predictor(u) @ predictor(i).T`, 64-d, ID only | `user_predictor(u) @ predictor(encoder(cat(i, fused))).T`, 128-d, includes modality |
| New config keys | `cl_weight` | `simsiam_weight`, `simsiam_hidden_dim`, `simsiam_output_dim`. `feat_embed_dim` is actually read here. In BM3 it is ignored. |
| Parameters, Baby (19,445 users / 7,050 items) | 33,570,688 (logged) | 33,739,968 (computed, +169,280, +0.50%) |
| Parameters, Sports (35,598 / 18,357) | 85,983,488 (logged) | 86,152,768 (logged, +169,280, +0.20%) |

## 4. What was replaced

| BM3 piece (location) | BM3SimSiam counterpart (location) |
|---|---|
| Shared `self.predictor = nn.Linear(64,64)` (bm3.py:49, 52) | `self.predictor = SimSiamPredictor(128→64→128)` (bm3_simsiam.py:113), now preceded by `self.encoder = SimSiamEncoder(192→256→128)` (line 106), plus `self.user_predictor = nn.Linear(64,128)` (line 120) for users |
| `image_trs` / `text_trs` as separate Linear layers (bm3.py:56, 60) | `MultiModalFusion.image_proj` / `text_proj`, whose outputs are concatenated (bm3_simsiam.py:87; simsiam_modules.py) |
| `forward` returns `i_g + h` (bm3.py:100) | `forward` returns `cat([i_g, fused], -1)` (bm3_simsiam.py:246) |
| Dropout target views `u/i/t/v_target` (bm3.py:111-124) | `z.detach()` as target in `calculate_simsiam_loss` (bm3_simsiam.py:275) |
| `loss_ui`, `loss_iu` with detached dropout targets (bm3.py:148-149) | `calculate_recommendation_loss`, a one-directional `1 - cos` (bm3_simsiam.py:282-311) |
| `loss_t`, `loss_v` (L_align) and `loss_tv`, `loss_vt` (L_ic) (bm3.py:139-146) | **No counterpart.** A single item-only SimSiam loss takes their place (bm3_simsiam.py:252-280). |
| `cl_weight` (bm3.py:36, 152; BM3.yaml:8) | `simsiam_weight` (bm3_simsiam.py:51; BM3SimSiam.yaml:10) |
| `reg_loss(u_online_ori, i_online_ori)` over full propagated matrices (bm3.py:151) | `reg_loss(user_embedding[users], item_id_embedding[items])`, batch only (bm3_simsiam.py:342-345) |
| `full_sort_predict`: shared predictor, 64-d (bm3.py:154-159) | `full_sort_predict`: `user_predictor` and `encoder` + `predictor`, 128-d (bm3_simsiam.py:356-383) |
| Config: `cl_weight` | Config: `simsiam_weight`, `simsiam_hidden_dim`, `simsiam_output_dim` |

## 5. Unchanged parts

- `get_norm_adj_mat`: the code is identical (bm3.py:62-89, bm3_simsiam.py:126-173).
- LightGCN propagation and mean pooling over layers: bm3.py:91-99 equals bm3_simsiam.py:224-241.
- `user_embedding` and `item_id_embedding`, with the same init.
- Pretrained `text_embedding` / `image_embedding` (trainable).
- The 384→64 and 4096→64 linear projections (location moved), with the same Xavier normal init.
- `EmbLoss` itself (only its arguments changed), `use_neg_sampling: False`, `embedding_size` 64, `n_layers` 2, `reg_weight` 0.1, `dropout` value 0.3.
- Trainer (`src/common/trainer.py`): `model.train()` for `calculate_loss`, and `model.eval()` under `no_grad` for `full_sort_predict` (trainer.py:141, 289-302). Optimizer is Adam with lr 0.001.
- Ranking is a dot product with no normalization. It is trained with cosine in both models, so the train/score mismatch exists in both.
- Data loading, split and evaluator.

### Parameter count derivation (d=64, text dim 384, image dim 4096)

- Shared by both: user and item ID embeddings + text and image embeddings + projections. Baby: 1,244,480 + 451,200 + 2,707,200 + 28,876,800 + 24,640 + 262,208 = 33,566,528.
- BM3 adds the predictor Linear(64,64) = 4,160. Total 33,570,688, which equals the logged "Trainable parameters" (`src/training_output.log:114`).
- BM3SimSiam adds the encoder 148,480, the predictor 16,640 and the user predictor 8,320 = 173,440. Total 33,739,968. The doc table (BM3_SIMSIAM.md) gives 33.74M, which agrees.
- The difference, 173,440 − 4,160 = 169,280, is independent of the dataset. On Sports, my formula gives 85,983,488 for BM3 and 86,152,768 for BM3SimSiam. Both match the logged counts in `BM3_results_summary.txt` and in `BM3SimSiam_results_summary.txt` (the latter only in commit `9e540d7`, via `git show`).
- The image embedding table (n_items × 4096) makes up about 86% of Baby parameters and about 87% of Sports parameters. It is the same in both models, which is why the models are nearly identical in size.

## 6. Doc claims verified or refuted (`docs/BM3_SIMSIAM.md`)

| Claim | Verdict | Evidence |
|---|---|---|
| BM3 uses a momentum-updated target network | **Refuted** | No EMA code in bm3.py. The target is a dropout clone (111-124). The doc's own snippet ("Key Innovation") shows dropout, which contradicts the momentum claim. |
| "Memory 2× model parameters" for BM3 | **Refuted** | BM3 has a single network. Its logged parameters are 33.57M. |
| BM3SimSiam "removes momentum" | **Vacuous** | There was none to remove. What was removed is dropout views, the symmetric detached losses and the modal losses. |
| SimSiam target is `z.detach()`, parameter-free | Verified | bm3_simsiam.py:275 |
| BM3 uses "simple linear projection", BM3SimSiam a "3-layer encoder" | **Misleading** | The same linear projections remain (in `MultiModalFusion`). The encoder is added on top of the concatenated item vector. |
| "Features fused early, processed together" and the diagram showing fusion feeding LightGCN | **Partly wrong** | Fusion output is concatenated after propagation (bm3_simsiam.py:246). Only ID embeddings are propagated. |
| "dropout: minimal usage" | **Understated** | `self.dropout` is never used (line 48 only). |
| Loss `L_reg = (‖u‖² + ‖i‖²)/batch` | **Inaccurate** | `EmbLoss` uses unsquared norms (`torch.norm(p=2)`), divided by batch size (loss.py:46-52). |
| Config uses grid values, n_layers default 1, dropout default 0.1, 1000 epochs, early stop 20 | **Stale** | The current YAML has single values (n_layers 2, dropout 0.3), and overall.yaml has epochs 75 and `stopping_step` 10. |
| Benchmark table (BM3 0.0883 / SimSiam TBD) | **Stale** | `COMPARISON_BM3_vs_BM3SimSiam.md` reports Sports, 600 epochs: 0.0765 vs 0.0869. The 75-epoch summaries show 0.0372 val vs 0.0842 val. These come from different datasets than the doc's Baby setup. |
| Parameters 33.57M vs 33.74M | Verified | Section 5 |

Note: `BM3_results_summary.txt` and `BM3SimSiam_results_summary.txt` contain pasted prompt text (a "Modify the provided Markdown..." task description) in the middle of the "Trainable Parameters" line. The numbers can still be read, but the files are corrupted.

## 7. Bugs and inconsistencies in `bm3_simsiam.py`

1. **No stop-gradient and no negatives in `loss_rec`** (lines 296-311). Minimizing `1 - cos(user_pred, item_pred)` has a degenerate solution: both heads output the same constant direction. The user and item predictors have biases, and the BN on the encoder does not prevent the predictor head from outputting a constant. BM3 avoided this with detached, dropout-perturbed targets. This is the likeliest cause of the weaker accuracy and slow convergence (Recall@20 about 0.037 at 75 epochs). It is untested.
2. **The SimSiam loss is trivially satisfiable** (lines 268-278). The target is the encoder output for the same input, so the predictor can learn the identity map. SimSiam relies on two augmented views. The real SimSiam setup is not reproduced here. The docs call it "contrastive", but nothing is contrasted. The report in COMPARISON_BM3_vs_BM3SimSiam.md correctly calls it a "single-view stop-gradient variant".
3. **Lost the BM3 modality objectives.** The multimodal alignment losses (L_align, L_ic) are gone. Text and image features now influence the model only through the item input vector, with no explicit alignment with the ID or collaborative embeddings.
4. **BN train/eval mismatch.** Training BN statistics come from 2048 interaction-sampled items (popular items repeat). `evaluate()` calls `model.eval()` (trainer.py:289), so the encoder uses running averages. Scoring is over all n_items with a different distribution. The only "momentum" in BM3SimSiam is BN's running-stat momentum (default 0.1). This could hurt ranking quality; it is untested.
5. **Regularization is weaker and different.** BM3 regularizes all propagated embeddings with a divisor of n_items (7,050 on Baby, 18,357 on Sports). BM3SimSiam regularizes batch raw embeddings with a divisor of about 2,048. The same `reg_weight` 0.1 therefore does not mean the same strength in the two models. This weakens the fairness of the comparison.
6. **Unused or dead code:** `self.dropout` (line 48), `item_features` (lines 217/219), `import os` (line 5), and `SimSiamLoss` in common/loss.py. The `feat_embed_dim` value in BM3.yaml is ignored by BM3 (bm3.py:33).
7. **Misleading docstrings:** the forward docstring says LightGCN propagates the combined item features, and says "residual connection" at line 245 where the code concatenates. The class docstring says momentum is replaced, which is not true.
8. **Encoder runs twice on the same batch per step** (lines 304 and 269), which doubles the BN running-stat updates and the compute.
9. **Latent edge case:** `BatchNorm1d` in training raises an error for a batch of one row. This is not triggered today (Baby: 118,551 train rows is 57.9 batches of 2048, so the last batch has 1,815 rows; Sports: the last batch has 1,321 rows). It would break on other datasets.
10. **Tests:** `src/test_bm3simsiam.py` and `src/test_simsiam_integration.py` exist but were not run here because PyTorch is missing.

## 8. Conclusions and recommendations (nothing implemented)

- Describe the change as "BM3's dropout-bootstrap loss head replaced by a single-view SimSiam head, with modality features moved from the loss into the item representation", not as "momentum target replaced by stop-gradient".
- Correct `docs/BM3_SIMSIAM.md`, `README.md` (line 51) and the `bm3_simsiam.py` class docstring: remove all momentum/EMA claims, the "2× memory" row, and the "dropout minimal usage" wording, and fix the diagram so the fusion output bypasses LightGCN.
- To test hypothesis 1 (collapse), add an ablation of `loss_rec` with `item_pred.detach()` or the symmetric BM3 form, and a variant that restores `cl_weight`-style modal alignment. Compare against BM3 with identical epochs and `stopping_step`.
- To make the comparison fair, restore `EmbLoss` on the full propagated embeddings (or rescale `reg_weight`), and use the same training length for both.
- Consider a true two-view SimSiam: pass two dropout views of the item input through the encoder and predictor, and apply the symmetric stop-gradient loss.
- Remove dead code (`item_features`, unused `dropout`, duplicate encoder call). Restore the original BM3 grid in `BM3.yaml` or store the single-value configs separately.
- Run the existing tests in an environment with PyTorch, and re-measure parameter counts with `print(model)` for confirmation.
