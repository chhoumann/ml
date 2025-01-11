import torch
from torch.utils.data import DataLoader

from build_a_large_language_model_from_scratch.lib.generate import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)
from build_a_large_language_model_from_scratch.lib.loss import (
    calc_loss_batch,
    calc_loss_loader,
)


def train_model_simple(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # reset loss gradients from previous iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # calculate loss gradients
            optimizer.step()  # update model weights using the loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(
    model, train_loader: DataLoader, val_loader: DataLoader, device, eval_iter: int
):
    model.eval()  # to disable dropout during evaluation
    with torch.no_grad():  # to disable gradient tracking, it's not required (reduce computational overhead)
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
