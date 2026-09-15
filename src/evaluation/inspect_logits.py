"""Inspect every next-token logit and choose generation tokens in a local browser.

Run: uv run python src/evaluation/inspect_logits.py MODEL.pt tokenizer.model
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import sentencepiece as spm
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pavullmo"))
from generate import load_model


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logit inspector</title>
<style>
  body { font: 16px system-ui, sans-serif; max-width: 1500px; margin: 2rem auto; padding: 0 1.5rem; }
  #layout { display: grid; grid-template-columns: minmax(320px, 2fr) minmax(650px, 3fr); gap: 2rem; align-items: start; }
  #left { position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow: auto; }
  textarea { width: 100%; box-sizing: border-box; font: inherit; }
  button { cursor: pointer; }
  #text { display: flex; flex-wrap: wrap; gap: .4rem; background: #f4f4f4; padding: 1rem; min-height: 2rem; }
  #text button { border: 1px solid #ccc; border-radius: 999px; background: white; padding: .35rem .65rem; font: 14px ui-monospace, monospace; overflow-wrap: anywhere; }
  #text button:hover { background: #e7efff; }
  #status { min-height: 1.5rem; }
  table { border-collapse: collapse; table-layout: fixed; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; border-bottom: 1px solid #ddd; padding: .35rem .5rem; }
  th:nth-child(1) { width: 3rem; }
  th:nth-child(3) { width: 4.5rem; }
  th:nth-child(4) { width: 5.5rem; }
  th:nth-child(5) { width: 35%; }
  td button { font: inherit; text-align: left; max-width: 100%; overflow-wrap: anywhere; }
  .bar-track { height: .8rem; background: #eee; border-radius: 999px; }
  .bar-fill { height: 100%; background: #578cd2; border-radius: inherit; }
</style>
<h1>Logit inspector</h1>
<div id="layout">
<section id="left">
<label for="prompt">Prompt</label>
<textarea id="prompt" rows="4" placeholder="Enter a prompt (or leave empty for BOS)"></textarea>
<p><button id="inspect">Inspect / reset</button></p>
<h2>Decoded text</h2>
<p>Each pill is one tokenizer piece (▁ marks a space). Click a pill to continue from that token.</p>
<div id="text"></div>
<p id="status">Enter a prompt, then click Inspect / reset.</p>
</section>
<section id="right">
<h2>Next-token logits</h2>
<p>Click a token to add exactly that token and inspect the next step. All vocabulary tokens, including special tokens, are available.</p>
<p>Bar width is the token's softmax probability (100% means full width).</p>
<table><thead><tr><th>Rank</th><th>Token</th><th>ID</th><th>Logit</th><th>Bar</th></tr></thead><tbody id="tokens"></tbody></table>
<p><button id="more" hidden>Show next 100</button></p>
</section>
</div>
<script>
const promptBox = document.getElementById('prompt');
const inspectButton = document.getElementById('inspect');
const moreButton = document.getElementById('more');
const status = document.getElementById('status');
const text = document.getElementById('text');
const table = document.getElementById('tokens');
let sequenceIds = [];
let candidates = [];
let shown = 0;
let requestNumber = 0;
let maxLogit = 0;
let softmaxDenominator = 1;

function clearCandidates() {
  candidates = [];
  shown = 0;
  table.replaceChildren();
  moreButton.hidden = true;
}

function showSequence(ids, pieces) {
  const pills = document.createDocumentFragment();
  pieces.forEach((piece, index) => {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.textContent = piece;
    pill.title = 'Continue after token ' + ids[index];
    pill.addEventListener('click', () => step(ids.slice(0, index + 1)));
    pills.append(pill);
  });
  text.replaceChildren(pills);
}

function showMore() {
  const end = Math.min(shown + 100, candidates.length);
  const rows = document.createDocumentFragment();
  for (let i = shown; i < end; i++) {
    const [id, piece, logit] = candidates[i];
    const row = document.createElement('tr');
    const rank = row.insertCell();
    rank.textContent = String(i + 1);
    const tokenCell = row.insertCell();
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = piece;
    button.title = 'Choose token ' + id;
    button.addEventListener('click', () => step([...sequenceIds, id]));
    tokenCell.append(button);
    row.insertCell().textContent = String(id);
    row.insertCell().textContent = logit.toFixed(5);
    const track = document.createElement('div');
    track.className = 'bar-track';
    const bar = document.createElement('div');
    bar.className = 'bar-fill';
    const probability = Math.exp(logit - maxLogit) / softmaxDenominator;
    bar.style.width = (100 * probability) + '%';
    track.title = `Softmax probability: ${(100 * probability).toPrecision(4)}%`;
    track.append(bar);
    row.insertCell().append(track);
    rows.append(row);
  }
  table.append(rows);
  shown = end;
  moreButton.hidden = shown >= candidates.length;
}

async function step(nextIds) {
  const request = ++requestNumber;
  inspectButton.disabled = true;
  clearCandidates();
  status.textContent = 'Computing logits…';
  try {
    const response = await fetch('/api/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(nextIds === null
        ? {prompt: promptBox.value}
        : {token_ids: nextIds}),
    });
    const result = await response.json();
    if (request !== requestNumber) return;
    if (!response.ok) throw new Error(result.error || 'Request failed');
    sequenceIds = result.token_ids;
    showSequence(result.token_ids, result.pieces);
    status.textContent = result.done
      ? result.reason
      : `Step ${sequenceIds.length} · ${result.context_used}/${result.context_limit} context tokens · ${result.tokens.length} candidates`;
    candidates = result.tokens || [];
    maxLogit = candidates.length ? candidates[0][2] : 0;
    softmaxDenominator = candidates.reduce(
      (total, candidate) => total + Math.exp(candidate[2] - maxLogit), 0);
    showMore();
  } catch (error) {
    if (request === requestNumber) status.textContent = 'Error: ' + error.message;
  } finally {
    if (request === requestNumber) inspectButton.disabled = false;
  }
}

promptBox.addEventListener('input', () => {
  requestNumber++;
  sequenceIds = [];
  clearCandidates();
  text.replaceChildren();
  inspectButton.disabled = false;
  status.textContent = 'Prompt changed. Click Inspect / reset.';
});
inspectButton.addEventListener('click', () => {
  step(null);
});
moreButton.addEventListener('click', showMore);
</script>
</html>
""".encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the model checkpoint")
    parser.add_argument("tokenizer", type=Path, help="path to the tokenizer model")
    parser.add_argument("--port", type=int, default=8000, help="localhost port (default: 8000)")
    return parser.parse_args()


def make_handler(model, tokenizer, device, vocab_size, sequence_length):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code: int, value: dict) -> None:
            self.send_bytes(
                code,
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_bytes(200, PAGE, "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/step":
                self.send_error(404)
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("expected application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 1_000_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                if "token_ids" in request:
                    token_ids = request["token_ids"]
                    if (
                        not isinstance(token_ids, list)
                        or not 1 <= len(token_ids) <= sequence_length
                        or token_ids[0] != tokenizer.bos_id()
                        or any(
                            type(token_id) is not int
                            or not 0 <= token_id < vocab_size
                            for token_id in token_ids
                        )
                    ):
                        raise ValueError("token_ids must be a valid sequence starting with BOS")
                else:
                    prompt = request.get("prompt")
                    if not isinstance(prompt, str):
                        raise ValueError("prompt must be text")
                    token_ids = [tokenizer.bos_id(), *tokenizer.encode(prompt, out_type=int)]
                    if len(token_ids) >= sequence_length:
                        raise ValueError(
                            f"prompt is too long for the {sequence_length}-token context"
                        )
                if tokenizer.eos_id() in token_ids[:-1]:
                    raise ValueError("generation already ended at EOS")

                result = {
                    "token_ids": token_ids,
                    "pieces": [tokenizer.id_to_piece(token_id) for token_id in token_ids],
                    "context_used": len(token_ids),
                    "context_limit": sequence_length,
                }
                if token_ids[-1] == tokenizer.eos_id():
                    result.update(done=True, reason="EOS selected. Generation ended.")
                elif len(token_ids) == sequence_length:
                    result.update(done=True, reason="Model context is full. Reset the prompt to continue.")
                else:
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and torch.cuda.is_bf16_supported()
                        else nullcontext()
                    )
                    with torch.inference_mode(), autocast:
                        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
                        logits = model(input_ids)[0, -1].float().cpu()
                    scores = logits.tolist()
                    ranked_ids = torch.argsort(logits, descending=True).tolist()
                    result.update(
                        done=False,
                        tokens=[
                            [token_id, tokenizer.id_to_piece(token_id), scores[token_id]]
                            for token_id in ranked_ids
                        ],
                    )
                self.send_json(200, result)
            except (ValueError, TypeError) as error:
                self.send_json(400, {"error": str(error)})

    return Handler


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model, vocab_size, sequence_length = load_model(args.model, device)
    if tokenizer.vocab_size() != vocab_size:
        raise ValueError(
            f"tokenizer vocabulary size {tokenizer.vocab_size()} does not match "
            f"checkpoint vocabulary size {vocab_size}"
        )
    if tokenizer.bos_id() < 0 or tokenizer.eos_id() < 0:
        raise ValueError("tokenizer must define both BOS and EOS tokens")

    server = HTTPServer(
        ("127.0.0.1", args.port),
        make_handler(model, tokenizer, device, vocab_size, sequence_length),
    )
    print(f"Loaded {args.model} on {device}.")
    print(f"Open http://127.0.0.1:{server.server_port}/ (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
