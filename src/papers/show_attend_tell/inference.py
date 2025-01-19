import argparse
import pickle

import torch
from PIL import Image
from torchvision import transforms as T

from papers.show_attend_tell.data.build_vocab import Vocabulary  # noqa: F401
from papers.show_attend_tell.models.decoder import DecoderWithAttention
from papers.show_attend_tell.models.encoder import EncoderCNN


def load_checkpoint(encoder, decoder, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])


def greedy_caption(image, encoder, decoder, vocab, device, max_len=20):
    """
    image: preprocessed PIL image or tensor -> (3,H,W)
    """
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(image, Image.Image):
        transformed_image = transform(image)
        image = transformed_image.unsqueeze(0).to(device)
    else:
        image = image.to(device)

    encoder_out = encoder(image)  # (1, encoder_dim, num_pixels)
    # Permute to (batch_size, num_pixels, encoder_dim)
    encoder_out = encoder_out.permute(0, 2, 1)

    # Init hidden and cell
    h, c = decoder.init_hidden_states(encoder_out)

    # Start token
    start_token = vocab.stoi["<SOS>"]
    end_token = vocab.stoi["<EOS>"]

    word_idx = start_token
    caption_indices = [word_idx]

    for _ in range(max_len):
        embeddings = decoder.embedding(
            torch.tensor([word_idx]).to(device)
        )  # (1, embed_dim)
        context, alpha = decoder.attention(encoder_out, h)  # (1, 2048), (1, num_pixels)
        lstm_input = torch.cat([embeddings, context], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))

        output = decoder.fc(h)  # (1, vocab_size)
        _, predicted = output.max(dim=1)  # greedy
        word_idx = predicted.item()
        caption_indices.append(word_idx)

        if word_idx == end_token:
            break

    # Convert indices to words
    words = [vocab.itos[idx] for idx in caption_indices]
    return " ".join(words[1:-1])  # remove <SOS> and <EOS> in string form


def beam_search_caption(
    image, encoder, decoder, vocab, device, beam_size=3, max_len=20
):
    """
    Implements beam search for better caption generation.

    Args:
        image: preprocessed PIL image or tensor -> (3,H,W)
        beam_size: number of best hypotheses to keep at each step
        max_len: maximum length of the caption
    Returns:
        best_caption: string of best caption
    """
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(image, Image.Image):
        transformed_image = transform(image)
        image = transformed_image.unsqueeze(0).to(device)
    else:
        image = image.to(device)

    # Encode image
    encoder_out = encoder(image)  # (1, encoder_dim, num_pixels)
    encoder_out = encoder_out.permute(0, 2, 1)  # (1, num_pixels, encoder_dim)

    # Initialize LSTM state
    h, c = decoder.init_hidden_states(encoder_out)

    # We'll treat the problem as having a batch size of k
    k = beam_size

    # Expand encoder_out, h, and c to match beam size
    encoder_out = encoder_out.expand(
        k, *encoder_out.shape[1:]
    )  # (k, num_pixels, encoder_dim)
    h = h.expand(k, *h.shape[1:])  # (k, decoder_dim)
    c = c.expand(k, *c.shape[1:])  # (k, decoder_dim)

    # Tensor to store top k previous words at each step
    seqs = torch.full((k, 1), vocab.stoi["<SOS>"], dtype=torch.long, device=device)

    # Tensor to store top k sequences' scores
    top_k_scores = torch.zeros(k, 1).to(device)

    # Lists to store completed sequences and scores
    complete_seqs = []
    complete_seqs_scores = []

    step = 1
    while True:
        embeddings = decoder.embedding(seqs[:, -1])  # (k, embed_dim)
        context, _ = decoder.attention(
            encoder_out, h
        )  # (k, encoder_dim), (k, num_pixels)

        lstm_input = torch.cat([embeddings, context], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))  # (k, decoder_dim)

        scores = decoder.fc(h)  # (k, vocab_size)
        scores = torch.log_softmax(scores, dim=1)

        # Add previous scores
        scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, dim=0)  # (k)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  # (k)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // len(vocab)  # (k)
        next_word_inds = top_k_words % len(vocab)  # (k)

        # Add new words to sequences
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
        )  # (k, step+1)

        # Which sequences are incomplete (didn't reach <EOS>)?
        incomplete_inds = [
            ind
            for ind, next_word in enumerate(next_word_inds)
            if next_word != vocab.stoi["<EOS>"]
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

        k = len(incomplete_inds)
        if k == 0:
            break

        # Update variables for incomplete sequences
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > max_len:
            break
        step += 1

    # If no complete sequences, use partial sequences
    if not complete_seqs:
        complete_seqs = seqs.tolist()
        complete_seqs_scores = top_k_scores.tolist()

    # Find sequence with best score
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    # Convert indices to words
    words = [vocab.itos[idx] for idx in seq]
    return " ".join(words[1:-1])  # remove < SOS > and <EOS>


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Init models
    encoder = EncoderCNN(encoded_image_size=14).to(device)
    decoder = DecoderWithAttention(
        attention_dim=512,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(vocab),
        encoder_dim=encoder.encoder_dim,
    ).to(device)

    load_checkpoint(encoder, decoder, args.ckpt_path, device)
    encoder.eval()
    decoder.eval()

    # Load image
    image = Image.open(args.image_path).convert("RGB")

    if args.beam_size > 1:
        caption = beam_search_caption(
            image, encoder, decoder, vocab, device, beam_size=args.beam_size, max_len=30
        )
    else:
        caption = greedy_caption(image, encoder, decoder, vocab, device, max_len=30)
    print("Predicted Caption:", caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size for caption generation. Use 1 for greedy search.",
    )
    args = parser.parse_args()
    main(args)
