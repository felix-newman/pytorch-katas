"""CPU-scale nanoGPT speedrun: dense vs factorized vs sparse-basis embeddings.

Train a small modded GPT on WikiText-2 and compare sample efficiency
(val loss vs tokens) plus how much rare-token embeddings move when the
token itself is rarely updated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_katas.nanogpt.data import load_wikitext
from pytorch_katas.nanogpt.model import GPT, GPTConfig
from pytorch_katas.settings import DATA_DIR

LOG_DIR = DATA_DIR / "nanogpt" / "logs"
PLOT_DIR = Path(__file__).resolve().parents[3] / "notebooks" / "nanogpt"


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)


@torch.no_grad()
def estimate_loss(model: GPT, data: np.ndarray, batch_size: int, block_size: int, device: torch.device, iters: int) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def embedding_drift(init: torch.Tensor, current: torch.Tensor, freqs: np.ndarray) -> dict[str, float]:
    delta = (current - init).norm(dim=-1).cpu()
    order = np.argsort(freqs)
    n = max(len(order) // 10, 1)
    rare = order[:n]
    frequent = order[-n:]
    return {
        "mean_all": float(delta.mean()),
        "mean_rare": float(delta[rare].mean()),
        "mean_frequent": float(delta[frequent].mean()),
        "rare_to_frequent": float(delta[rare].mean() / (delta[frequent].mean() + 1e-8)),
    }


def cosine_lr(step: int, max_steps: int, lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return 0.1 * lr + 0.9 * lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_one(args: argparse.Namespace, embedding_type: str, dataset: dict, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = GPTConfig(
        vocab_size=dataset["vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        embedding_type=embedding_type,  # type: ignore[arg-type]
        n_basis=args.n_basis,
        k_sparse=args.k_sparse,
    )
    model = GPT(config).to(device)
    init_emb = model.token_embeddings().detach().cpu().clone()
    optimizers = model.configure_optimizers(args.lr, args.weight_decay)
    tokens_per_step = args.batch_size * args.block_size
    history: list[dict] = []
    t0 = time.perf_counter()
    reached_at: dict[str, float | int | None] = {"tokens": None, "seconds": None, "step": None}

    print(f"\n=== {embedding_type} | params={model.get_num_params():,} | vocab={config.vocab_size} ===", flush=True)
    for step in range(args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.lr, args.warmup)
        for opt in optimizers:
            for group in opt.param_groups:
                base = args.lr * 10.0 if isinstance(opt, torch.optim.Muon) else args.lr
                group["lr"] = lr / args.lr * base

        if step % args.eval_interval == 0 or step == args.max_steps:
            val = estimate_loss(model, dataset["val"], args.batch_size, args.block_size, device, args.eval_iters)
            elapsed = time.perf_counter() - t0
            tokens = step * tokens_per_step
            row = {"step": step, "tokens": tokens, "val_loss": val, "seconds": elapsed}
            history.append(row)
            print(f"step {step:4d}  tokens {tokens:8d}  val {val:.4f}  {elapsed:.1f}s", flush=True)
            if reached_at["tokens"] is None and val <= args.target_loss:
                reached_at = {"tokens": tokens, "seconds": elapsed, "step": step}

        if step == args.max_steps:
            break

        x, y = get_batch(dataset["train"], args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

    with torch.no_grad():
        drift = embedding_drift(init_emb, model.token_embeddings().detach().cpu(), dataset["freqs"])
    result = {
        "embedding_type": embedding_type,
        "params": model.get_num_params(),
        "history": history,
        "drift": drift,
        "reached_target": reached_at,
        "final_val": history[-1]["val_loss"],
        "min_val": min(r["val_loss"] for r in history),
        "seconds": history[-1]["seconds"],
    }
    print(f"final val {result['final_val']:.4f}  min val {result['min_val']:.4f}  rare/freq drift {drift['rare_to_frequent']:.3f}", flush=True)
    return result


def plot_results(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for result in results:
        hist = result["history"]
        axes[0].plot([r["tokens"] / 1e3 for r in hist], [r["val_loss"] for r in hist], marker="o", label=result["embedding_type"])
    axes[0].set_xlabel("tokens (thousands)")
    axes[0].set_ylabel("WikiText-2 val loss")
    axes[0].set_title("Sample efficiency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    names = [r["embedding_type"] for r in results]
    rare = [r["drift"]["mean_rare"] for r in results]
    freq = [r["drift"]["mean_frequent"] for r in results]
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width / 2, rare, width, label="rare tokens")
    axes[1].bar(x + width / 2, freq, width, label="frequent tokens")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("L2 embedding movement from init")
    axes[1].set_title("Does the basis move unused/rare tokens?")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sparse-basis nanoGPT speedrun")
    p.add_argument("--embedding", choices=["dense", "factorized", "sparse", "all"], default="all")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-basis", type=int, default=64)
    p.add_argument("--k-sparse", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--target-loss", type=float, default=6.0)
    p.add_argument("--min-freq", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--smoke", action="store_true", help="tiny run for CI / sanity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_steps = 20
        args.eval_interval = 10
        args.eval_iters = 2
        args.batch_size = 8
        args.block_size = 32
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    dataset = load_wikitext(min_freq=args.min_freq)
    print(f"train tokens={len(dataset['train']):,} val tokens={len(dataset['val']):,} vocab={dataset['vocab_size']}")

    kinds = ["dense", "factorized", "sparse"] if args.embedding == "all" else [args.embedding]
    results = [run_one(args, kind, dataset, device) for kind in kinds]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = "smoke" if args.smoke else "speedrun"
    log_path = LOG_DIR / f"{stamp}.json"
    log_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    plot_path = PLOT_DIR / f"{stamp}_curves.png"
    plot_results(results, plot_path)
    print(f"wrote {log_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
