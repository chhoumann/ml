import os
import pickle

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from papers.show_attend_tell.data.build_vocab import Vocabulary  # noqa: F401


class CaptionDataset(Dataset):
    def __init__(
        self, image_dir, caption_file, vocab_path, split="train", transform=None
    ):
        """
        image_dir: directory with Flickr8k images
        caption_file: path to captions.txt file
        vocab_path: pickled Vocabulary object
        split: one of ['train', 'val', 'test']
        transform: optional image transform pipeline
        """
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.split = split

        # Load annotations from Flickr8k format
        df = pd.read_csv(caption_file)

        # Load split information
        splits = {"train": 0.7, "val": 0.15, "test": 0.15}

        # Get unique image names and their counts
        unique_images = df["image"].unique()
        n_images = len(unique_images)

        # Calculate split indices
        train_idx = int(n_images * splits["train"])
        val_idx = train_idx + int(n_images * splits["val"])

        # Split images
        if split == "train":
            split_images = unique_images[:train_idx]
        elif split == "val":
            split_images = unique_images[train_idx:val_idx]
        else:  # test
            split_images = unique_images[val_idx:]

        # Filter annotations based on split
        self.annotations = [
            {"file_name": row["image"], "caption": row["caption"]}
            for _, row in df.iterrows()
            if row["image"] in split_images
        ]

        # Load vocab
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        caption = ann["caption"]
        file_name = ann["file_name"]  # or build from ann["image_id"]

        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Numericalize caption
        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(caption)
        tokens.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(tokens)


def caption_collate_fn(batch):
    """
    Pads captions to the max length in the batch.
    Returns:
      images: stacked tensor of shape (B, C, H, W)
      targets: padded captions (B, max_length)
      lengths: original lengths of each caption
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True)  # sort by caption length desc

    images, captions = zip(*batch)  # separate image and caption

    images = torch.stack(images, dim=0)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]

    return images, padded_captions, lengths


def get_loader(
    image_dir,
    caption_file,
    vocab_path,
    transform,
    split="train",
    batch_size=32,
    shuffle=True,
    num_workers=4,
):
    dataset = CaptionDataset(
        image_dir=image_dir,
        caption_file=caption_file,
        vocab_path=vocab_path,
        split=split,
        transform=transform,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=caption_collate_fn,
        pin_memory=True,
    )
    return loader


# Example usage:
if __name__ == "__main__":
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_path = "src/papers/show_attend_tell"

    # Create train, val, and test loaders
    train_loader = get_loader(
        image_dir=f"{base_path}/data/Images",
        caption_file=f"{base_path}/data/captions.txt",
        vocab_path=f"{base_path}/data/vocab.pkl",
        transform=transform,
        split="train",
    )

    val_loader = get_loader(
        image_dir=f"{base_path}/data/Images",
        caption_file=f"{base_path}/data/captions.txt",
        vocab_path=f"{base_path}/data/vocab.pkl",
        transform=transform,
        split="val",
        shuffle=False,
    )

    test_loader = get_loader(
        image_dir=f"{base_path}/data/Images",
        caption_file=f"{base_path}/data/captions.txt",
        vocab_path=f"{base_path}/data/vocab.pkl",
        transform=transform,
        split="test",
        shuffle=False,
    )

    # Print sizes of each split
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
