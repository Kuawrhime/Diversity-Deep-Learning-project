# Skin Cancer Detection with ResNet-18

> A ResNet-18 binary classifier achieves test ROC-AUC = 0.902613 on ISIC 2019 benign-vs-malignant skin lesion classification.

## Setup

```bash
git clone https://github.com/Kuawrhime/Diversity-Deep-Learning-project.git
cd Diversity-Deep-Learning-project
pip install -r requirements.txt
```

## Reproducing Our Headline Result

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

Expected output:

```text
test_roc_auc = 0.902613
test_accuracy = 0.786316
test_balanced_accuracy = 0.804804
```

To recompute metrics from images instead of printing the recorded headline metrics, download the ISIC 2019 training data and place it at:

```text
ISIC_2019_Training_Input/
ISIC_2019_Training_Input/ISIC_2019_Training_GroundTruth.csv
```

Dataset source: https://challenge.isic-archive.com/data/#2019

Then run:

```bash
python train.py --config configs/default.yaml
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## Demo

Run inference on one skin lesion image:

```bash
python sample.py --checkpoint checkpoints/best.pt --image path/to/skin_lesion.jpg
```

The script prints the predicted class and malignant probability.

## Hardware + Training Time

Training used one CUDA GPU with seed 42 pinned in `configs/default.yaml`. The notebook run used 15 epochs with batch size 128 and a 70/15/15 stratified train/validation/test split.

## Authors + License

Project authors: Karim MOHAMED, Laurent ZHANG, Corentin ALBERTUS, Arnaud AUGAIT, Romain ZHANG

MIT License.

## AI Disclosure

ChatGPT was used to write the README, and create configuration/result metadata.
