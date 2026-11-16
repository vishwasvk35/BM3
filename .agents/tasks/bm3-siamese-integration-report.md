# BM3 and Siamese/SimSiam Integration Findings

## Summary answer

1. **Standard BM3 is implemented.** `src/models/bm3.py` defines `BM3`, with user/item embeddings, LightGCN-style propagation over the interaction graph, text/image feature projections, prediction head, and a multi-term bootstrap/contrastive loss.
2. **A BM3SimSiam variant is implemented and connected to the normal training path.** `src/models/bm3_simsiam.py` defines `BM3SimSiam`; `src/utils/utils.py:get_model()` maps the CLI name `BM3SimSiam` to that module; and `src/main.py` passes `-m/--model` to `utils.quick_start.quick_start()`. Its dedicated YAML configuration is present at `src/configs/model/BM3SimSiam.yaml`. Running from `src` with `python main.py -d sports -m BM3SimSiam` would therefore instantiate it through the shared trainer.
3. **The variant is BM3-inspired with SimSiam components, not a faithful two-branch Siamese/SimSiam implementation.** It retains BM3's ID-based LightGCN and multi-modal recommendation setting, adds early fusion of text and image features, a three-layer encoder, a two-layer predictor, and a detached encoder target. However, its self-supervised loss compares `predictor(encoder(x))` with `detach(encoder(x))` for the *same* item representation. The code creates neither two augmented views nor two encoder branches. It is accurate to call it a stop-gradient SimSiam-style modification, but not a conventional Siamese-pair SimSiam training objective.
4. **Repository evidence supports that the variant was trained at least once, but the cited Sports checkpoints are absent.** Committed summaries and a 75-epoch SimSiam log report Sports results. The `.pth` files named in those reports are ignored and are not present in this checkout. The only available checkpoint is the standard-BM3 Baby model at `models/trained/baby/baby/BM3-Oct-20-2026-12-11-13.pth`.

## Evidence

### Standard BM3 implementation

- **Model:** `src/models/bm3.py:BM3`
  - `__init__` creates trainable user/item embeddings, trainable preloaded text/image embeddings, linear text/image projections, a predictor, and a normalized bipartite adjacency matrix.
  - `forward()` performs `n_layers` sparse graph propagations and averages the embeddings across layers, then returns graph-enhanced users and items.
  - `calculate_loss()` uses dropout-noised detached copies of online user/item/text/image representations as targets. It optimizes user-to-item and item-to-user cosine objectives plus text/image alignment and embedding regularization.
  - `full_sort_predict()` produces all-item scores for shared evaluation.
- **Configuration:** `src/configs/model/BM3.yaml` configures two graph layers, `dropout: 0.3`, `reg_weight: 0.1`, and `cl_weight: 2.0`; `src/configs/overall.yaml` sets the 75-epoch training/evaluation defaults.
- **Existing artifact:** `models/trained/baby/baby/BM3-Oct-20-2026-12-11-13.pth` is the only checkpoint found in the working tree.

### BM3SimSiam implementation

- **Model:** `src/models/bm3_simsiam.py:BM3SimSiam`
  - `__init__` builds the same style of normalized user-item graph, user/item IDs, trainable text/image embeddings, `MultiModalFusion`, `SimSiamEncoder`, `SimSiamPredictor`, and a user prediction projection.
  - `get_fused_item_features()` projects and concatenates available text and image features.
  - `forward()` runs LightGCN on ID embeddings, then concatenates the graph-propagated item ID embedding with the fused item features. Despite one code comment and documentation diagram, the fused features do **not** pass through LightGCN; only IDs do.
  - `calculate_recommendation_loss()` aligns the user projection with the encoded-and-predicted positive item projection.
  - `calculate_simsiam_loss()` computes `z = encoder(embeddings)`, `p = predictor(z)`, `z_target = z.detach()`, and negative cosine similarity between `p` and `z_target`.
  - `calculate_loss()` adds recommendation loss, `simsiam_weight *` the stop-gradient loss, and regularization. This proves the SimSiam-style objective is part of training rather than an unused helper.
- **Reusable modules:** `src/models/simsiam_modules.py`
  - `SimSiamEncoder` is a three-layer MLP with BatchNorm and ReLU in the first two layers.
  - `SimSiamPredictor` is a two-layer asymmetric MLP with BatchNorm/ReLU in its hidden layer.
  - `MultiModalFusion` separately projects text and image inputs to `feat_embed_dim` and concatenates them.
- **Configuration:** `src/configs/model/BM3SimSiam.yaml` supplies `simsiam_weight: 2.0`, `simsiam_hidden_dim: 64`, and `simsiam_output_dim: 128` alongside the BM3-style graph and regularization settings.
- **Dispatch and training integration:**
  - `src/main.py` accepts `-m BM3SimSiam` and calls `quick_start(model=args.model, ...)`.
  - `src/utils/quick_start.py:quick_start()` constructs the selected class with `get_model(config['model'])(config, train_data)`, then trains it using the shared `Trainer`.
  - `src/utils/utils.py:get_model()` explicitly converts the lowercased `BM3SimSiam` name to `models.bm3_simsiam`, imports it, and returns the `BM3SimSiam` class.
  - `src/utils/configurator.py:Config._load_dataset_model_config()` loads `configs/model/<model>.yaml`, so it resolves `BM3SimSiam.yaml` when started from `src`, as the README commands require.

### Tests, docs, and outputs

- **Test scripts:**
  - `src/test_simsiam_integration.py` exercises fusion dimensions, encoder/predictor shapes, stop-gradient behavior, a backward pass, and an end-to-end synthetic loss path.
  - `src/test_bm3simsiam.py` loads the Baby dataset, resolves the model via `get_model`, then attempts model construction, forward pass, loss/backward pass, and full-sort prediction.
  - Both are standalone print-oriented scripts with no assertions or test-runner configuration. They are useful smoke checks, but they do not provide strong automated verification. They were not run in this read-only investigation because the task prohibited expensive training and the dataset-based test is resource-intensive.
- **Documentation:** `README.md` and `docs/BM3_SIMSIAM.md` explicitly document `python main.py -d baby -m BM3SimSiam`. `COMPARISON.md` describes the claimed architecture and a Sports comparison. These documents correctly identify the added encoder, predictor, fusion, and stop-gradient concepts, but some details conflict with source code, noted below.
- **Recorded results:**
  - `BM3SimSiam_training_log_75epochs.txt` contains a CPU Sports training run for `model=BM3SimSiam`, the exact configuration in `BM3SimSiam.yaml`, a printed 86,152,768 trainable-parameter model, per-epoch metrics, and checkpoint-save messages.
  - `BM3SimSiam_results_summary.txt` reports a best Sports validation Recall@20 of 0.0372 at epoch 69 and test Recall@20 of 0.0366.
  - `BM3_results_summary.txt` reports Sports BM3 test Recall@20 of 0.0869 at epoch 73.
  - `COMPARISON_BM3_vs_BM3SimSiam.txt` and `COMPARISON.md` report the same comparison, with BM3's Recall@20 about 2.37 times the SimSiam variant's (0.0869 versus 0.0366).
  - The reports cite `src/saved/BM3-sports-Nov-16-2026-11-57-43.pth` and `src/saved/BM3SimSiam-sports-Nov-16-2026-09-43-24.pth`, but `src/saved/` does not exist in this checkout. `.gitignore` excludes `src/saved/*.pth` and all `*.pth`, explaining why those claims cannot be independently evaluated here.

## Conclusions and recommendations

### Conclusion

The repository does combine a working standard BM3 implementation with an integrated **BM3SimSiam** experimental variant. The two choices can be selected through one CLI and use the same data-loading, training, evaluation, and checkpoint mechanisms. It is not one model that runs BM3 and a generic Siamese network side by side; it is a separate BM3-derived model class that replaces BM3's feature-processing/loss path with fusion plus encoder/predictor and a stop-gradient objective.

The source code qualifies the terminology: `BM3SimSiam` contains SimSiam-inspired machinery, but it lacks the two differently augmented views and symmetric cross-view comparison that define conventional SimSiam. Its self-supervised branch aligns a prediction with a detached representation of the same input. Any scientific or product claim should describe it as a **SimSiam-style stop-gradient variant**, unless the objective is revised to use two views.

### Recommended fixes, without implementation

1. **Align documentation with source.** `README.md`, `docs/BM3_SIMSIAM.md`, and `COMPARISON.md` say standard BM3 has a momentum-updated target network. `BM3.calculate_loss()` instead uses cloned, detached representations with dropout; it defines no target-network parameters or exponential-moving-average update. Document the implemented behavior or implement an actual momentum target before making that claim.
2. **Make the variant a genuine SimSiam objective if that is the intended research contribution.** Create two item representation views, apply encoder/predictor in the expected cross-view direction, detach only the target side, and test gradient flow across both views. Otherwise rename/descriptively document the current loss as a single-view stop-gradient predictor objective.
3. **Correct the fusion/LightGCN architecture diagram and prose.** `BM3SimSiam.forward()` propagates user and item ID embeddings through LightGCN, then appends fused multimodal item features; it does not propagate the fused 192-dimensional representation through the graph. The local `item_features` variable is also calculated but unused.
4. **Strengthen reproducibility.** Add assertion-based unit tests for module shapes and gradient boundaries, a small CPU fixture for full model integration, and a test command. Preserve or publish checksums/download locations for the Sports checkpoints and associate summaries with their exact commit/configuration. Current numerical claims are supported by committed text logs, but cannot be reproduced or checkpoint-validated from this checkout.
5. **Remove or use `common.loss:SimSiamLoss`.** It was added with the integration commit, but `BM3SimSiam` directly calls `F.cosine_similarity` instead of this helper. One canonical loss implementation reduces the risk of documentation and behavior drifting apart.

## Investigation scope

This was a read-only source-and-artifact review. No training, checkpoint evaluation, or test scripts were run. The repository was at commit `9e540d7` (`Added Siamese Network`) during inspection; that commit added the SimSiam model, modules, configuration, tests, documentation, result summaries, and model dispatch support.
