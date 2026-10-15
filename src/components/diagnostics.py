"""Shared parameter grouping for training diagnostics and offline analysis."""


def gradient_group_name(parameter_name: str) -> str:
    if parameter_name == "embeddings.weight":
        return "embeddings"
    for pattern, group in (
        (".attn.c_attn.", "attention_qkv"),
        (".attn.q_proj.", "attention_query"),
        (".attn.k_proj.", "attention_key"),
        (".attn.v_proj.", "attention_value"),
        (".attn.c_proj.", "attention_output"),
        (".ffn.w1.", "ffn_input_gate"),
        (".ffn.w3.", "ffn_input_gate"),
        (".ffn.w2.", "ffn_output"),
    ):
        if pattern in parameter_name:
            return group
    if "_norm." in parameter_name or parameter_name == "norm.weight":
        return "norms"
    return "other"
