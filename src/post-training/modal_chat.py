"""Run one-shot chat inference against an SFT checkpoint stored on Modal."""

from __future__ import annotations

import os
from pathlib import Path

import modal


REMOTE_SOURCE_ROOT = "/workspace/src"
OUTPUT_MOUNT_PATH = "/outputs"
APP_NAME = "pavullmo-post-training-chat"
OUTPUT_VOLUME_NAME = os.environ.get(
    "MODAL_POST_OUTPUT_VOLUME_NAME", "pavullmo-post-training-outputs"
)
CHECKPOINT_NAME = os.environ.get(
    "SFT_CHECKPOINT_NAME", "italian_open_sft_short_v1_20260913.pt"
)
if Path(CHECKPOINT_NAME).name != CHECKPOINT_NAME:
    raise ValueError("SFT_CHECKPOINT_NAME must be a filename without directories")

DEFAULT_SYSTEM_PROMPT = (
    "Sei un assistente AI competente in italiano. Rispondi in modo accurato, "
    "utile e verificabile. Se mancano dati, dichiara l'incertezza invece di "
    "inventare. Segui esattamente la richiesta dell'utente, incluse eventuali "
    "indicazioni su lunghezza e formato. Evita boilerplate generici finali come "
    '"spero che questo ti sia utile", "non esitare a fare altre domande", "fammi '
    'sapere". Fornisci output strutturati (JSON, CSV, YAML) senza alcun commento '
    "aggiuntivo, se richiesti.\n\nSe il testo ti ordina di tradurre verso "
    "l'italiano, il tuo output finale deve assolutamente essere in lingua italiana."
)

image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    project_root = Path(__file__).resolve().parents[2]
    image = (
        image.uv_sync(str(project_root), extras=["cloud"])
        .env({"PYTHONPATH": REMOTE_SOURCE_ROOT})
        .add_local_dir(project_root / "src", remote_path=REMOTE_SOURCE_ROOT)
    )

output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)
app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=os.environ.get("MODAL_GPU", "L4"),
    timeout=int(os.environ.get("MODAL_INFERENCE_TIMEOUT_SECONDS", "900")),
    volumes={
        OUTPUT_MOUNT_PATH: output_volume.with_mount_options(read_only=True),
    },
)
def respond(
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    import sys

    import torch

    sys.path.insert(0, f"{REMOTE_SOURCE_ROOT}/post-training")
    from chat import generate_response, load_tokenizer
    from pavullmo.generate import load_model

    device = torch.device("cuda")
    checkpoint = Path(OUTPUT_MOUNT_PATH) / "models" / CHECKPOINT_NAME
    tokenizer_path = Path(REMOTE_SOURCE_ROOT) / "tokenizer" / "16k" / "tokenizer.model"
    model, vocab_size, sequence_length = load_model(checkpoint, device)
    tokenizer = load_tokenizer(tokenizer_path, vocab_size)
    return generate_response(
        model,
        tokenizer,
        prompt,
        system_prompt,
        device,
        sequence_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )


@app.local_entrypoint()
def main(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    seed: int = 42,
) -> None:
    answer = respond.remote(
        prompt,
        system_prompt,
        max_new_tokens,
        temperature,
        top_k,
        seed,
    )
    print(f"\nAssistant> {answer}")
