import numpy as np
import torch


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # iterative over transformer blocks
    for b in range(len(params["blocks"])):
        # split is used to divide attention and bias weights into three equal parts for the qkv components
        # load attention qkv weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].layers[0][1].W_query.weight = assign(
            gpt.trf_blocks[b].layers[0][1].W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].layers[0][1].W_key.weight = assign(
            gpt.trf_blocks[b].layers[0][1].W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].layers[0][1].W_value.weight = assign(
            gpt.trf_blocks[b].layers[0][1].W_value.weight, v_w.T
        )

        # load attn qkv bias
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].layers[0][1].W_query.bias = assign(
            gpt.trf_blocks[b].layers[0][1].W_query.bias, q_b
        )
        gpt.trf_blocks[b].layers[0][1].W_key.bias = assign(
            gpt.trf_blocks[b].layers[0][1].W_key.bias, k_b
        )
        gpt.trf_blocks[b].layers[0][1].W_value.bias = assign(
            gpt.trf_blocks[b].layers[0][1].W_value.bias, v_b
        )

        # load attn linear projection weights
        gpt.trf_blocks[b].layers[0][1].out_proj.weight = assign(
            gpt.trf_blocks[b].layers[0][1].out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].layers[0][1].out_proj.bias = assign(
            gpt.trf_blocks[b].layers[0][1].out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # load feedforward network weights and biases
        gpt.trf_blocks[b].layers[1][1].layers[0].weight = assign(
            gpt.trf_blocks[b].layers[1][1].layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].layers[1][1].layers[0].bias = assign(
            gpt.trf_blocks[b].layers[1][1].layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt.trf_blocks[b].layers[1][1].layers[2].weight = assign(
            gpt.trf_blocks[b].layers[1][1].layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].layers[1][1].layers[2].bias = assign(
            gpt.trf_blocks[b].layers[1][1].layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # load layer norm params
        gpt.trf_blocks[b].layers[0][0].scale = assign(
            gpt.trf_blocks[b].layers[0][0].scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].layers[0][0].shift = assign(
            gpt.trf_blocks[b].layers[0][0].shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].layers[1][0].scale = assign(
            gpt.trf_blocks[b].layers[1][0].scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].layers[1][0].shift = assign(
            gpt.trf_blocks[b].layers[1][0].shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # Original GPT-2 model reused the token embedding weights to reduce the total number of params (weight tying)
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def assign(left, right):
    """Assigns values from right tensor to left tensor after shape validation.

    Args:
        left: Target PyTorch tensor/parameter
        right: Source tensor/array to copy values from

    Returns:
        torch.nn.Parameter: New parameter containing values from right tensor

    Raises:
        ValueError: If shapes of left and right tensors don't match
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    return torch.nn.Parameter(torch.tensor(right))
