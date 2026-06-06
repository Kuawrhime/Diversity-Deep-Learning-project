import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def build_image_index(image_dir):
    image_index = {}
    if not os.path.exists(image_dir):
        return image_index

    for root, _, files in os.walk(image_dir):
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                base_name = os.path.splitext(file_name)[0]
                full_path = os.path.join(root, file_name)
                image_index[base_name] = full_path
                image_index[file_name] = full_path
    return image_index


def find_image_path(img_name, image_index):
    if img_name in image_index:
        return image_index[img_name]
    base_name = os.path.splitext(img_name)[0]
    if base_name in image_index:
        return image_index[base_name]
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        if img_name + ext in image_index:
            return image_index[img_name + ext]
        if base_name + ext in image_index:
            return image_index[base_name + ext]
    return None


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_index, image_size, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row["image"]
        img_path = find_image_path(img_name, self.image_index)
        if img_path is None:
            image = Image.new("RGB", (self.image_size, self.image_size), color="black")
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (self.image_size, self.image_size), color="black")

        if self.transform:
            image = self.transform(image)
        return image, int(row["binary_label"]), img_name


def get_transforms(image_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def prepare_dataframe(config):
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["csv_path"])
    if "UNK" in df.columns:
        df = df[df["UNK"] != 1]

    target_classes = data_cfg["benign_classes"] + data_cfg["malignant_classes"]
    available_classes = [c for c in target_classes if c in df.columns]
    if not available_classes:
        raise ValueError("No configured class columns were found in the CSV.")
    df = df[df[available_classes].sum(axis=1) == 1].copy()

    malignant = set(data_cfg["malignant_classes"])

    def get_binary_label(row):
        return int(any(row.get(cls, 0) == 1 for cls in malignant))

    df["binary_label"] = df.apply(get_binary_label, axis=1)
    image_index = build_image_index(data_cfg["image_dir"])
    df["image_exists"] = df["image"].apply(lambda x: find_image_path(x, image_index) is not None)
    df = df[df["image_exists"]].drop(columns=["image_exists"]).copy()
    if df.empty:
        raise ValueError("No labeled rows have matching image files.")
    return df, image_index


def split_dataframe(df, config):
    data_cfg = config["data"]
    seed = config["project"]["seed"]
    train_val_df, test_df = train_test_split(
        df,
        test_size=data_cfg["test_ratio"],
        random_state=seed,
        stratify=df["binary_label"],
    )
    val_ratio_adjusted = data_cfg["val_ratio"] / (data_cfg["train_ratio"] + data_cfg["val_ratio"])
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=train_val_df["binary_label"],
    )
    return train_df, val_df, test_df


def make_loaders(config):
    df, image_index = prepare_dataframe(config)
    train_df, val_df, test_df = split_dataframe(df, config)
    image_size = config["model"]["image_size"]
    train_transform, eval_transform = get_transforms(image_size)

    train_dataset = SkinLesionDataset(train_df, image_index, image_size, train_transform)
    val_dataset = SkinLesionDataset(val_df, image_index, image_size, eval_transform)
    test_dataset = SkinLesionDataset(test_df, image_index, image_size, eval_transform)

    class_counts = [sum(train_df["binary_label"] == 0), sum(train_df["binary_label"] == 1)]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = torch.tensor(
        [class_weights[int(label)].item() for label in train_df["binary_label"]],
        dtype=torch.float,
    )
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    loader_cfg = config["training"]
    train_loader = DataLoader(train_dataset, batch_size=loader_cfg["batch_size"],
                              sampler=sampler, num_workers=loader_cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=loader_cfg["batch_size"],
                            shuffle=False, num_workers=loader_cfg["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=loader_cfg["batch_size"],
                             shuffle=False, num_workers=loader_cfg["num_workers"])
    return train_loader, val_loader, test_loader, class_counts
