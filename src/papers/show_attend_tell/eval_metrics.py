import nltk


def compute_bleu(candidate_sentence, reference_sentences):
    """
    candidate_sentence: list of tokens
    reference_sentences: list of lists of tokens
    """
    # e.g., references = [reference_sentences] if you have 1 reference
    # candidate and references must be lists of tokens (strings)
    return nltk.translate.bleu_score.sentence_bleu(
        reference_sentences, candidate_sentence
    )
