# BM3 vs. BM3SimSiam: Experimental Results on the Sports Dataset
 
**Document type:** Experimental results report  
**Status:** Training of BM3SimSiam completed through 600 epochs (interrupted due to technical issues; planned schedule was 1,000 epochs).

## 1. Purpose

This report presents a comparative evaluation of BM3 and BM3SimSiam on the Sports dataset. BM3SimSiam is a BM3-inspired graph-based recommendation model that incorporates multimodal item features together with a SimSiam-style encoder, predictor, and stop-gradient objective. It is implemented as a single-view stop-gradient variant rather than a conventional two-view Siamese SimSiam architecture. The primary goal is to quantify the performance of BM3SimSiam after extended training relative to the BM3 baseline.

## 2. Experimental Setup

### 2.1 Dataset

| Item | Value |
| --- | ---: |
| Dataset | Sports |
| Number of users | 35,598 |
| Number of items | 18,357 |

All experiments used the same dataset split and evaluation protocol.

### 2.2 Reference Performance

| Item | Value |
| --- | ---: |
| BM3 test Recall@20 | 0.0869 |
| BM3SimSiam test Recall@20 after 75 epochs | 0.0366 |
| BM3SimSiam Recall@20 at 75 epochs relative to BM3 | −57.9% |

At 75 epochs, BM3 outperformed BM3SimSiam across all 16 evaluated metrics. The substantial gap observed at this early stage motivated continuation of BM3SimSiam training.

### 2.3 Training Configuration

BM3SimSiam was trained for a planned total of 1,000 epochs. Training was interrupted at epoch 600 because of technical issues. All reported results for BM3SimSiam therefore correspond to the checkpoint obtained after 600 epochs of training. The same evaluation configuration, metrics, and data split employed for the BM3 baseline were used throughout.

## 3. Model Architecture

| Aspect | BM3 | BM3SimSiam |
| --- | --- | --- |
| Self-supervision mechanism | Momentum-based target network | Stop-gradient objective with predictor |
| Encoder | Standard BM3 architecture | SimSiam-style encoder (3 layers) |
| View structure | BM3 formulation | Single-view stop-gradient variant |
| Parameters | 85.98M | 86.15M |

BM3SimSiam retains the core graph-based recommendation structure of BM3 while replacing the momentum encoder with a stop-gradient formulation that includes an explicit predictor network. Parameter counts remain nearly identical (0.2% difference).

## 4. Training Progress

BM3SimSiam exhibited steady improvement from the 75-epoch checkpoint to the 600-epoch checkpoint. Recall@20 rose from 0.0366 to 0.0765, corresponding to an absolute gain of 0.0399 (+109.0%). Training was terminated at epoch 600 due to technical issues before the planned 1,000-epoch schedule could be completed. Intermediate checkpoints were retained for analysis but are not reported in full here.

## 5. Evaluation Results (Test Set, 600 Epochs)

### 5.1 Recall

| Metric | BM3 | BM3SimSiam (600 epochs) | Difference |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.0345 | 0.0304 | −0.0041 |
| Recall@10 | 0.0551 | 0.0485 | −0.0066 |
| Recall@20 | 0.0869 | 0.0765 | −0.0104 |
| Recall@50 | 0.1467 | 0.1291 | −0.0176 |

### 5.2 NDCG

| Metric | BM3 | BM3SimSiam (600 epochs) | Difference |
| --- | ---: | ---: | ---: |
| NDCG@5 | 0.0234 | 0.0206 | −0.0028 |
| NDCG@10 | 0.0301 | 0.0265 | −0.0036 |
| NDCG@20 | 0.0383 | 0.0337 | −0.0046 |
| NDCG@50 | 0.0504 | 0.0444 | −0.0060 |

### 5.3 Precision

| Metric | BM3 | BM3SimSiam (600 epochs) | Difference |
| --- | ---: | ---: | ---: |
| Precision@5 | 0.0076 | 0.0067 | −0.0009 |
| Precision@10 | 0.0061 | 0.0054 | −0.0007 |
| Precision@20 | 0.0048 | 0.0042 | −0.0006 |
| Precision@50 | 0.0033 | 0.0029 | −0.0004 |

### 5.4 MAP

| Metric | BM3 | BM3SimSiam (600 epochs) | Difference |
| --- | ---: | ---: | ---: |
| MAP@5 | 0.0191 | 0.0168 | −0.0023 |
| MAP@10 | 0.0218 | 0.0192 | −0.0026 |
| MAP@20 | 0.0239 | 0.0210 | −0.0029 |
| MAP@50 | 0.0258 | 0.0227 | −0.0031 |

## 6. Improvement of BM3SimSiam from 75 to 600 Epochs

| Metric | Measured at 75 epochs | Measured at 600 epochs | Absolute gain |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.0124 | 0.0304 | +0.0180 |
| Recall@10 | 0.0212 | 0.0485 | +0.0273 |
| Recall@20 | 0.0366 | 0.0765 | +0.0399 |
| Recall@50 | 0.0709 | 0.1291 | +0.0582 |
| NDCG@5 | 0.0076 | 0.0206 | +0.0130 |
| NDCG@10 | 0.0105 | 0.0265 | +0.0160 |
| NDCG@20 | 0.0145 | 0.0337 | +0.0192 |
| NDCG@50 | 0.0214 | 0.0444 | +0.0230 |
| Precision@5 | 0.0028 | 0.0067 | +0.0039 |
| Precision@10 | 0.0024 | 0.0054 | +0.0030 |
| Precision@20 | 0.0020 | 0.0042 | +0.0022 |
| Precision@50 | 0.0016 | 0.0029 | +0.0013 |
| MAP@5 | 0.0058 | 0.0168 | +0.0110 |
| MAP@10 | 0.0069 | 0.0192 | +0.0123 |
| MAP@20 | 0.0080 | 0.0210 | +0.0130 |
| MAP@50 | 0.0090 | 0.0227 | +0.0137 |

## 7. Comparative Analysis

### 7.1 Primary Metric

| Training point | BM3SimSiam Recall@20 | Gap to BM3 (0.0869) | Relative change from BM3 | Status |
| --- | ---: | ---: | ---: | --- |
| 75 epochs | 0.0366 | −0.0503 | −57.9% | Measured |
| 600 epochs | 0.0765 | −0.0104 | −12.0% | Measured |

After 600 epochs, BM3SimSiam reduced the relative gap on Recall@20 from 57.9% to 12.0%. Absolute performance remains below the BM3 baseline on every reported metric.

### 7.2 Summary of Relative Performance

Across the full suite of 16 metrics, BM3SimSiam at 600 epochs trails BM3 by approximately 12%. The absolute differences are consistent in direction and scale with the primary Recall@20 result. BM3 retains a clear advantage on all cut-offs of Recall, NDCG, Precision, and MAP.

## 8. Interpretation of Results

Extended training produced substantial gains for BM3SimSiam. The model more than doubled its Recall@20 between epoch 75 and epoch 600. Nevertheless, the final performance remains 12.0% below BM3 on the primary metric and exhibits a comparable relative deficit on the remaining metrics. The stop-gradient formulation with predictor appears capable of continued improvement under longer schedules, yet it has not closed the gap to the momentum-based BM3 baseline within the 600 epochs completed.

The near-identical parameter counts indicate that the observed difference is unlikely to be explained by model capacity. Differences in the self-supervision mechanism (stop-gradient versus momentum target network) and the single-view design remain the most plausible sources of the residual performance gap.

## 9. Limitations

- Training was interrupted at epoch 600 because of technical issues; the planned 1,000-epoch schedule was not completed. Further improvement beyond 600 epochs cannot be ruled out.
- Results are reported from a single training run. No repeated trials, alternative random seeds, or confidence intervals are available.
- Early-stopping behavior under the original patience setting was not re-evaluated for the longer schedule.
- No additional hyper-parameter search was conducted for the extended training regime.
- The evaluation is restricted to the Sports dataset; generalization to other domains is not assessed in this report.

## 10. Conclusion

BM3SimSiam trained for 600 epochs achieves a test Recall@20 of 0.0765, compared with 0.0869 for BM3. The absolute difference is −0.0104 (−12.0%). This represents a marked recovery from the 75-epoch result of 0.0366 (−57.9% relative to BM3). Across all 16 metrics the same pattern holds: BM3SimSiam improves substantially with additional training yet remains consistently below the BM3 baseline. Training was terminated at epoch 600 due to technical issues before the intended 1,000-epoch budget could be exhausted. The residual gap suggests that the stop-gradient, single-view formulation has not yet matched the effectiveness of BM3’s momentum-based self-supervision under the conditions examined.

## 11. Saved Models

| Model | Path |
| --- | --- |
| BM3 | `src/saved/BM3-sports-Nov-16-2026-11-57-43.pth` |
| BM3SimSiam (600 epochs) | `src/saved/BM3SimSiam-sports-Nov-16-2026-09-43-24.pth` |