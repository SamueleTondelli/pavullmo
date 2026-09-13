"""Shared validation and tokenization helpers for SFT conversations."""

from __future__ import annotations

from typing import Any

import sentencepiece as spm
import torch


ROLES = ("system", "user", "assistant")


def validate_messages(example_id: str, messages: list[object]) -> None:
    previous_role: str | None = None
    assistant_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{example_id!r} message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLES:
            raise ValueError(f"{example_id!r} message {index} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{example_id!r} message {index} has empty content")
        if role == "system" and index != 0:
            raise ValueError(f"{example_id!r} system message must be first")
        if role == "user" and previous_role not in {None, "system", "assistant"}:
            raise ValueError(f"{example_id!r} has consecutive user messages")
        if role == "assistant" and previous_role != "user":
            raise ValueError(f"{example_id!r} assistant message must follow a user message")
        assistant_count += role == "assistant"
        previous_role = role
    if previous_role != "assistant" or assistant_count == 0:
        raise ValueError(f"{example_id!r} must end with an assistant response")


def encode_example(
    example: dict[str, Any],
    tokenizer: spm.SentencePieceProcessor,
    context_length: int,
) -> dict[str, Any]:
    token_ids = [tokenizer.bos_id()]
    loss_mask = [False]
    text_parts: list[str] = []

    for message in example["messages"]:
        role = message["role"]
        content = message["content"].strip()
        role_id = tokenizer.piece_to_id(f"<{role}>")
        if role_id == tokenizer.unk_id():
            raise ValueError(f"tokenizer does not define <{role}>")
        content_ids = tokenizer.encode(f"\n{content}\n", out_type=int)
        learns_response = role == "assistant"
        token_ids.extend([role_id, *content_ids])
        loss_mask.extend([learns_response] * (1 + len(content_ids)))
        text_parts.append(f"<{role}>\n{content}")

    token_ids.append(tokenizer.eos_id())
    loss_mask.append(True)
    if len(token_ids) > context_length:
        raise ValueError(
            f"{example['id']!r} uses {len(token_ids)} tokens, exceeding "
            f"context length {context_length}"
        )
    if sum(loss_mask) < 2:
        raise ValueError(f"{example['id']!r} has no assistant targets")

    source_metadata = {
        key: value
        for key, value in example.items()
        if key not in {"id", "messages"}
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    return {
        "id": example["id"],
        "category": example.get("category", "unspecified"),
        "source_metadata": source_metadata,
        "text": "\n".join(text_parts),
        "token_ids": torch.tensor(token_ids, dtype=torch.int32),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
    }
