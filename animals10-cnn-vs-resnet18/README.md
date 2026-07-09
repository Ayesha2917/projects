# Animal Species Classification: Custom CNN vs Fine-Tuned ResNet18

A side-by-side comparison of two approaches to multi-class image classification on the **Animals-10** dataset: a Convolutional Neural Network trained **entirely from scratch**, versus a **ResNet18 fine-tuned** via transfer learning.

---

##  Dataset

- **Name:** [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) (Kaggle)
- **Classes (10):** dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel
- **Total images:** 26,179
- **Split:** 70% train / 15% val / 15% test → 18,325 / 3,926 / 3,928 images
- **Notes:** Naturally imbalanced (dog: 4,863 images vs. elephant: 1,446 images). Original folder names are in Italian; mapped to English class names in the notebook.

##  Models Compared

| | Custom CNN | ResNet18 (Fine-Tuned) |
|---|---|---|
| **Pretraining** | None — trained from scratch | ImageNet1K pretrained |
| **Architecture** | 4× (Conv → BatchNorm → ReLU → MaxPool) blocks + GAP + FC head | Standard ResNet18, `fc` layer replaced for 10 classes |
| **Input size** | 96×96 | 224×224 (ImageNet normalization) |
| **Freeze strategy** | N/A (100% trainable) | Frozen: `conv1, bn1, layer1, layer2, layer3` · Trainable: `layer4, fc` |
| **Total params** | 423,562 | 11,181,642 |
| **Trainable params** | 423,562 (100%) | 8,398,858 (75.1%) |
| **Epochs** | 15 | 10 |

##  Final Test Set Results

| Metric | Custom CNN | ResNet18 (Fine-Tuned) |
|---|---|---|
| **Accuracy** | 0.7235 | **0.9674** |
| **Precision** | 0.7431 | **0.9675** |
| **Recall** | 0.7235 | **0.9674** |
| **F1-score** | 0.7158 | **0.9674** |
| **Best Val Accuracy** | 0.7173 | **0.9633** |
| **Training Time** | 12.0 min | 16.0 min |
| **Inference Time (full test set)** | **7.48s** | 13.10s |

**Key takeaway:** ResNet18's ImageNet pretraining gives it a ~24-point accuracy edge over the from-scratch CNN, despite freezing 75% of its layers and training for fewer epochs. The custom CNN struggled most on visually similar or underrepresented classes (elephant: 37% recall, cat: 33% recall), while ResNet18 stayed above 93% recall across every class. The trade-off: ResNet18 has ~26× more parameters and roughly 1.75× the inference time.

### Per-class performance highlights
- **Custom CNN weak points:** elephant (54% F1), cat (45% F1), sheep (59% F1)
- **ResNet18:** consistently 93–99% F1 across all 10 classes

##  Repo Structure

```
animals10-cnn-vs-resnet18/
├── Animals10_CNN_vs_ResNet.ipynb   # Main notebook (data pipeline, training, evaluation, plots)
├── images/                          # Exported plots (training curves, confusion matrices, etc.)
├── requirements.txt
└── README.md
```

##  How to Run

1. Open the notebook in **Google Colab** (GPU runtime recommended: `Runtime > Change runtime type > T4 GPU`)
2. Get a Kaggle API token: **Kaggle → Account Settings → API → Create New Token**, upload `kaggle.json` when prompted
3. Run all cells top to bottom — dataset downloads automatically via the Kaggle API

##  What's Inside the Notebook

- Dataset download & class distribution analysis
- Separate preprocessing pipelines (96×96 for CNN, 224×224 + ImageNet norm for ResNet)
- Custom CNN architecture built from scratch
- Configurable layer-freezing for ResNet18 fine-tuning
- Shared training loop for a fair comparison
- Full evaluation: accuracy, precision, recall, F1, confusion matrices
- Training curves, bar chart comparisons, parameter/efficiency plots
- Single-image inference demo with both models side by side

##  Tech Stack

`PyTorch` · `torchvision` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn` · `Kaggle API`

---

*Part of a broader ML/AI project portfolio — see the [projects](../) root for more.*
