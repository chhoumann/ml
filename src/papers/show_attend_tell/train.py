import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from papers.show_attend_tell.data.build_vocab import Vocabulary  # noqa: F401
from papers.show_attend_tell.data.dataset import get_loader
from papers.show_attend_tell.models.decoder import DecoderWithAttention
from papers.show_attend_tell.models.encoder import EncoderCNN


def train_one_epoch(
    loader, encoder, decoder, criterion, optimizer, device, scaler, clip_value
):
    encoder.train()
    decoder.train()

    epoch_loss = 0.0

    for imgs, captions, lengths in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            encoder_out = encoder(imgs)  # (B, encoder_dim, num_pixels)
            preds, alphas = decoder(encoder_out, captions, lengths)

            # Shift targets by 1: compare preds[:, t] with captions[:, t+1]
            # preds: (B, max_len-1, vocab_size) - we predict up to second-last token
            # targets: (B, max_len-1) - we use tokens from index 1 onwards
            targets = captions[:, 1:]  # Remove first token (SOS)
            preds = preds[:, :-1, :]  # Remove last prediction

            # Flatten predictions and targets
            preds = preds.reshape(-1, preds.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(preds, targets)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip_value)
        nn.utils.clip_grad_norm_(encoder.parameters(), clip_value)

        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate_one_epoch(loader, encoder, decoder, criterion, device):
    encoder.eval()
    decoder.eval()

    epoch_val_loss = 0.0
    with torch.no_grad():
        for imgs, captions, lengths in tqdm(loader, desc="Validation", leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            encoder_out = encoder(imgs)
            preds, alphas = decoder(encoder_out, captions, lengths)

            # Shift targets by 1 as in training
            targets = captions[:, 1:]  # Remove first token (SOS)
            preds = preds[:, :-1, :]  # Remove last prediction

            preds = preds.reshape(-1, preds.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(preds, targets)
            epoch_val_loss += loss.item()

    return epoch_val_loss / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = get_loader(
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        vocab_path=args.vocab_path,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        split="train",
    )

    val_loader = get_loader(
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        vocab_path=args.vocab_path,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        split="val",
    )

    # Initialize models
    encoder = EncoderCNN(encoded_image_size=14).to(device)
    decoder = DecoderWithAttention(
        attention_dim=args.attention_dim,
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        vocab_size=args.vocab_size,
        encoder_dim=encoder.encoder_dim,
        dropout=args.dropout,
    ).to(device)

    # Optionally fine-tune the last few layers of encoder,
    # or freeze the encoder if you prefer.
    for param in encoder.resnet.parameters():
        param.requires_grad = args.finetune_encoder

    # Define loss and optimizer
    pad_idx = 0  # <PAD> token index
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    params = (
        list(decoder.parameters()) + list(encoder.parameters())
        if args.finetune_encoder
        else list(decoder.parameters())
    )
    optimizer = optim.AdamW(params, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    scaler = GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{args.num_epochs}]")
        train_loss = train_one_epoch(
            train_loader,
            encoder,
            decoder,
            criterion,
            optimizer,
            device,
            scaler,
            clip_value=args.clip,
        )
        print(f"Train Loss: {train_loss:.4f}")

        if val_loader:
            val_loss = validate_one_epoch(
                val_loader, encoder, decoder, criterion, device
            )
            print(f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder_state": encoder.state_dict(),
                        "decoder_state": decoder.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    os.path.join(args.save_dir, "best.ckpt"),
                )
                print("Saved Best Model!")

        scheduler.step()

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--caption_file", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
