import torch


# `idx` is a (batch, n_tokens) array of indices in the current context
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # crops current context if it exceeds supported context size (only last 'context_size' tokens are used as context if current context is larger than dontext_size)
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # focus on last time step
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat(
            (idx, idx_next), dim=1
        )  # appends sampled index to the running sequence. idx: (batch, n_tokens+1)

    return idx


def text_to_token_ids(text: str, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(
        0
    )  # `unsqueeze(0)` adds batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
