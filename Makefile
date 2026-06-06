.PHONY: train eval sample data

train:
	python train.py --config configs/default.yaml

eval:
	python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_resnet18_binary.pt

sample:
	python sample.py --config configs/default.yaml --checkpoint checkpoints/best_resnet18_binary.pt --image $(IMAGE)

data:
	python download_data.py
