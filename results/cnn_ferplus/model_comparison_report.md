# Convolutional facial expression experiment

The CNN receives aligned FER2013 48×48 grayscale face images and learns spatial filters over facial features. Training uses FER+ unique top-vote consensus labels (minimum four votes), with horizontal flips applied only to training batches. PublicTest selected the model; PrivateTest was evaluated once after selection.

Architecture: 3 convolution blocks (16,32,64), 3x3 conv per block, batch normalization, ReLU, max pooling, adaptive 3x3 pooling, 128-unit dense head; 48x48 grayscale input; train-only random horizontal flips

| Model | PublicTest consensus accuracy | Macro-F1 |
|---|---:|---:|
| Existing FER+ MLP | 0.5498 | 0.3636 |
| Spatial CNN + flip averaging | 0.7378 | 0.5159 |

Training stopped after 18 epochs (955.4s on cpu).
PrivateTest consensus: 0.7188 accuracy and 0.5081 macro-F1 on 3,329 accepted rows.

Per-class precision, recall, F1, and the confusion matrix are in `final_test_metrics_ferplus_consensus.json`. The candidate validation and training curves are in `validation_metrics.json` and `training_history.json`.
