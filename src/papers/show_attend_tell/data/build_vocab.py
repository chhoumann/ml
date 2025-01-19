import argparse
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd


class Vocabulary:
    def __init__(self, freq_threshold=5):
        """
        freq_threshold: minimum frequency to include a token in the vocab
        """
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index = 4

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        """
        sentence_list: list of all sentences in the dataset
        """
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = self.index
                self.itos[self.index] = word
                self.index += 1

    def numericalize(self, text):
        """
        Converts a text string into a list of token indices.
        """
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


def main(args):
    # Read the Flickr8k captions file
    captions_file = Path(args.dataset_path) / "captions.txt"
    df = pd.read_csv(captions_file, sep=",")

    # Extract all captions from the dataset
    # In Flickr8k, the 'caption' column contains the actual captions
    all_captions = df["caption"].tolist()

    # Create and build vocabulary
    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocabulary(all_captions)

    # Save vocab
    import pickle

    with open(args.vocab_output, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saved vocabulary to {args.vocab_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Flickr8k dataset directory",
    )
    parser.add_argument(
        "--freq_threshold",
        type=int,
        default=5,
        help="Minimum frequency threshold for words to be included in vocabulary",
    )
    parser.add_argument(
        "--vocab_output",
        type=str,
        default="vocab.pkl",
        help="Output path for the vocabulary pickle file",
    )
    args = parser.parse_args()
    main(args)
