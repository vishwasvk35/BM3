# BM3SimSiam Experimental Results Report

## Executive Summary

This report documents the available experimental comparison between the BM3 baseline and the BM3SimSiam variant using Recall@20. BM3SimSiam recorded Recall@20 of 0.0765 after 600 completed epochs. The corresponding recorded BM3 Recall@20 is 0.0869. Thus, at the reported BM3SimSiam training point, the variant is lower than BM3 by 0.0104 Recall@20, equivalent to a relative change of -12.0%.

BM3SimSiam training was scheduled for 1,000 epochs but was interrupted by technical issues at epoch 600. The results therefore describe the model at the interruption point. They do not represent a completed 1,000-epoch training run.

## Experimental Setup

### Evaluation Objective

The experiment compares BM3 and BM3SimSiam with Recall@20, which measures the proportion of relevant items recovered among the top 20 recommendations. The comparison is descriptive and is restricted to the reported values in this document.

| Experimental item | Value |
| --- | ---: |
| Baseline model | BM3 |
| Evaluated model | BM3SimSiam |
| Reported metric | Recall@20 |
| BM3 Recall@20 | 0.0869 |
| BM3SimSiam Recall@20 | 0.0765 |
| BM3SimSiam planned training budget | 1,000 epochs |
| BM3SimSiam completed training | 600 epochs |
| BM3SimSiam completion rate | 60% |
| Training outcome | Interrupted at epoch 600 because of technical issues |

### Recorded Baseline Context

The repository's BM3 results summary identifies the 0.0869 baseline as the Sports test Recall@20 from a 75-epoch BM3 run. That record reports a best validation epoch of 73. It also documents 35,598 users, 18,357 items, 296,337 interactions, and a 99.95% sparsity value for that BM3 reference run.

| BM3 reference-run field | Recorded value |
| --- | ---: |
| Dataset label | Sports |
| Maximum epochs | 75 |
| Best validation epoch | 73 |
| Test Recall@20 | 0.0869 |
| Users | 35,598 |
| Items | 18,357 |
| Interactions | 296,337 |
| Reported sparsity | 99.95% |

The supplied BM3SimSiam result is reported at 600 completed epochs. The available record does not establish that the BM3 reference run and the interrupted BM3SimSiam run used matching training budgets or otherwise identical experimental conditions. Accordingly, this report does not characterize the comparison as a controlled, matched-budget study.

## Model Architecture

BM3SimSiam is a BM3-inspired graph-based recommendation model that incorporates multimodal item features. User and item representations are learned in the user-item interaction graph, while item-side multimodal features contribute to the learned item representation. The architecture applies a SimSiam-style encoder and predictor, with a stop-gradient objective used to form the training target.

The objective is a single-view stop-gradient variant. It is not a conventional two-view Siamese SimSiam architecture. The encoder-predictor pathway refines representations within the recommendation model, and the stop-gradient operation prevents gradients from flowing through the target representation.

| Architectural aspect | BM3 | BM3SimSiam |
| --- | --- | --- |
| Recommendation backbone | Graph-based BM3 framework | BM3-inspired graph-based framework |
| Item information | Multimodal item features | Multimodal item features |
| Target construction | Momentum-based target mechanism in the recorded BM3 documentation | Stop-gradient target in the SimSiam-style objective |
| Representation refinement | BM3 projection-based learning | SimSiam-style encoder and predictor |
| SimSiam formulation | Not applicable | Single-view stop-gradient variant, not a conventional two-view Siamese design |

The architectural distinction motivates the comparison, but the reported metrics alone do not isolate the contribution of any individual component.

## Training Progress

BM3SimSiam was assigned a training budget of 1,000 epochs. Execution reached epoch 600 before technical issues interrupted training. The remaining 400 epochs were not completed.

| Training-progress measure | Value |
| --- | ---: |
| Planned epochs | 1,000 |
| Completed epochs | 600 |
| Uncompleted epochs | 400 |
| Completed fraction | 600 / 1,000 |
| Completion rate | 60% |
| Termination reason | Technical issues |

The reported BM3SimSiam Recall@20 of 0.0765 corresponds to the model state obtained after those 600 completed epochs. No conclusion is drawn about the Recall@20 that the model might have obtained had the remaining epochs been completed.

## Evaluation Results

Recall@20 is the sole BM3SimSiam evaluation metric supplied for the interrupted run. The table below presents the requested comparison without adding metrics that are not documented for that run.

| Model | Training status associated with reported result | Recall@20 | Absolute difference from BM3 | Relative performance change from BM3 |
| --- | --- | ---: | ---: | ---: |
| BM3 | Recorded baseline; 75-epoch reference run | 0.0869 | — | — |
| BM3SimSiam | Interrupted after 600 of 1,000 planned epochs | 0.0765 | -0.0104 | -12.0% |

For the reported values, BM3 has the higher Recall@20. BM3SimSiam reaches 0.0765, whereas BM3 reaches 0.0869.

## Comparative Analysis

The absolute Recall@20 difference is computed by subtracting the BM3 baseline from the BM3SimSiam result:

```text
BM3SimSiam Recall@20 - BM3 Recall@20
= 0.0765 - 0.0869
= -0.0104
```

The relative performance change is calculated with BM3 as the reference:

```text
(-0.0104 / 0.0869) × 100 = -11.97% ≈ -12.0%
```

The negative absolute and relative values indicate lower reported Recall@20 for BM3SimSiam at epoch 600. This comparison describes the observed values only. It does not attribute the difference to the stop-gradient objective, multimodal processing, encoder-predictor design, or any other individual architectural choice.

## Interpretation of Results

At the recorded training point, BM3SimSiam does not match the BM3 Recall@20 baseline. The observed gap is 0.0104 in absolute Recall@20 and -12.0% relative to BM3.

The interrupted schedule is material to the interpretation. BM3SimSiam completed 60% of its planned 1,000-epoch budget, so the result should be read as the performance measured at epoch 600 rather than as an endpoint for the planned training schedule. The available evidence supports comparison of the reported outcomes, but it does not establish the outcome of the uncompleted portion of training.

The two reported runs also have different documented training durations: the BM3 baseline record describes a 75-epoch run, while the BM3SimSiam result describes 600 completed epochs from a 1,000-epoch plan. Because matching experimental conditions for these reported values are not documented, the result should not be interpreted as a controlled attribution of performance to the architectural differences.

## Limitations

- BM3SimSiam training ended at epoch 600 because of technical issues, leaving 400 planned epochs uncompleted.
- The BM3SimSiam result supplied for this report contains Recall@20 only. No additional metrics, epoch-wise measurements, loss values, or parameter-sweep results are reported for the 600-epoch run.
- The available record does not provide the dataset configuration, data split, optimization settings, hardware, random seed, or repeated-trial information for the interrupted BM3SimSiam run.
- The BM3 baseline and BM3SimSiam result have different documented training durations. Matching conditions are not established by the available evidence.
- The reported values support neither reproducibility claims nor statistical-significance claims.

## Conclusion

BM3SimSiam completed 600 of its planned 1,000 epochs before technical issues interrupted training. At this recorded point, BM3SimSiam achieved Recall@20 of 0.0765. The recorded BM3 baseline achieved Recall@20 of 0.0869, producing an absolute difference of -0.0104 and a relative performance change of -12.0%.

BM3 therefore has the higher reported Recall@20 in this comparison. The result is limited to the observed BM3SimSiam state at epoch 600 and does not imply completion of the 1,000-epoch training plan or predict the outcome of the uncompleted epochs.
