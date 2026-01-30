# simulate_stream.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, Any, List, Tuple

import numpy as np

from baselines import MemoryItem, make_baseline
from memorysafe_buffer import MemorySafeBuffer


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class OnlineLogReg:
    """Tiny online logistic regression with SGD."""
    def __init__(self, d: int, lr: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=(d,))
        self.b = 0.0
        self.lr = float(lr)

    def predict_proba(self, x: np.ndarray) -> float:
        z = float(np.dot(self.w, x) + self.b)
        return sigmoid(z)

    def loss_and_grad(self, x: np.ndarray, y: int) -> Tuple[float, np.ndarray, float]:
        p = self.predict_proba(x)
        # logistic loss
        eps = 1e-8
        loss = -(y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps))
        # gradient
        dz = (p - y)
        gw = dz * x
        gb = dz
        grad_norm = float(np.linalg.norm(gw))
        return loss, gw, gb, grad_norm

    def step(self, gw: np.ndarray, gb: float):
        self.w -= self.lr * gw
        self.b -= self.lr * gb


def make_stream(step: int, d: int, rare_p: float, drift_every: int, seed: int, inject_noise: bool) -> Tuple[np.ndarray, int]:
    """
    Synthetic stream:
      - rare positives
      - concept drift: decision boundary rotates every drift_every steps
      - optional injected noisy samples (mislabeled / unstable)
    """
    rng = np.random.default_rng(seed + step)
    x = rng.normal(0, 1, size=(d,))

    # boundary drift: rotate a base vector
    t = (step // drift_every)
    base = np.zeros(d)
    base[0] = math.cos(0.25 * t)
    base[1] = math.sin(0.25 * t)
    base[2] = 0.5

    score = float(np.dot(base, x))
    # convert to probability, then sample label with rare skew
    p = sigmoid(score)
    y = 1 if (rng.random() < rare_p * (0.5 + p)) else 0

    if inject_noise:
        # inject periodic "bad memories": flip label and add extra feature noise
        if step % 500 == 0 and step > 0:
            y = 1 - y
            x = x + rng.normal(0, 2.5, size=(d,))
    return x, y


def eval_metrics(model: OnlineLogReg, xs: np.ndarray, ys: np.ndarray) -> Dict[str, float]:
    probs = np.array([model.predict_proba(x) for x in xs])
    preds = (probs >= 0.5).astype(int)

    tp = int(np.sum((preds == 1) & (ys == 1)))
    fp = int(np.sum((preds == 1) & (ys == 0)))
    fn = int(np.sum((preds == 0) & (ys == 1)))
    tn = int(np.sum((preds == 0) & (ys == 0)))

    recall_pos = tp / (tp + fn + 1e-8)
    prec_pos = tp / (tp + fp + 1e-8)
    f1_pos = 2 * prec_pos * recall_pos / (prec_pos + recall_pos + 1e-8)

    return {
        "recall_pos": float(recall_pos),
        "prec_pos": float(prec_pos),
        "f1_pos": float(f1_pos),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def ensure_runs_dir() -> str:
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"run_{stamp}.jsonl")
    # symlink-like convenience
    latest = os.path.join("runs", "latest.jsonl")
    try:
        if os.path.exists(latest):
            os.remove(latest)
        # copy-on-write: we just write to latest too
    except Exception:
        pass
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="memorysafe", choices=["memorysafe", "fifo", "reservoir", "random_replay"])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--capacity", type=int, default=800)
    ap.add_argument("--replay_k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--rare_p", type=float, default=0.06)
    ap.add_argument("--drift_every", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject_noise", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    model = OnlineLogReg(d=args.d, lr=args.lr, seed=args.seed)

    if args.policy == "memorysafe":
        buf = MemorySafeBuffer(
            capacity=args.capacity,
            seed=args.seed,
            mvi_forget_th=0.8,
            min_age_forget=250,
            protect_top_frac=0.15,
            forget_frac_when_full=0.06,
        )
    else:
        buf = make_baseline(args.policy, args.capacity, args.seed)

    # fixed eval set
    eval_xs = []
    eval_ys = []
    for i in range(1500):
        x, y = make_stream(i * 7 + 123, args.d, args.rare_p, args.drift_every, args.seed + 999, inject_noise=False)
        eval_xs.append(x)
        eval_ys.append(y)
    eval_xs = np.stack(eval_xs, axis=0)
    eval_ys = np.array(eval_ys, dtype=int)

    run_path = ensure_runs_dir()
    latest_path = os.path.join("runs", "latest.jsonl")

    # per-item tracking: utility EMA and loss variance approximations
    def update_item_stats(item: MemoryItem, loss: float, alpha: float = 0.05):
        u = float(item.meta.get("utility_ema", 0.0))
        # utility: higher if loss is high (the model needs this) AND if it's positive
        # this is a demo proxy; in real system you'd use contribution metrics
        target = min(1.0, loss / 2.0) + (0.4 if int(item.y) == 1 else 0.0)
        item.meta["utility_ema"] = (1 - alpha) * u + alpha * target

        # crude variance tracking
        m = float(item.meta.get("loss_ema", loss))
        v = float(item.meta.get("loss_var", 0.0))
        m_new = (1 - alpha) * m + alpha * loss
        v_new = (1 - alpha) * v + alpha * (loss - m_new) ** 2
        item.meta["loss_ema"] = m_new
        item.meta["loss_var"] = v_new

    def novelty_score(x: np.ndarray, buffer_items: List[MemoryItem]) -> float:
        if not buffer_items:
            return 1.0
        # distance to a small random subset of buffer points (cheap)
        subset = buffer_items[: min(25, len(buffer_items))]
        xs = np.array([it.x for it in subset], dtype=float)
        dists = np.linalg.norm(xs - x[None, :], axis=1)
        return float(np.clip(np.mean(dists) / 6.0, 0.0, 2.0))

    with open(run_path, "w", encoding="utf-8") as f_run, open(latest_path, "w", encoding="utf-8") as f_latest:
        for step in range(1, args.steps + 1):
            x, y = make_stream(step, args.d, args.rare_p, args.drift_every, args.seed, args.inject_noise)
            loss, gw, gb, gnorm = model.loss_and_grad(x, y)
            model.step(gw, gb)

            # add current item to buffer (with meta needed for governance)
            meta = {
                "loss": float(loss),
                "grad_norm": float(gnorm),
                "novelty": float(novelty_score(x, buf.items if hasattr(buf, "items") else [])),
                "volatility": 0.0,  # per-item vol tracked during replay updates
            }
            item = MemoryItem(x=x.tolist(), y=int(y), step=step, meta=meta)
            add_info = buf.add(item)

            # replay
            replay_items = buf.sample(args.replay_k)
            replay_loss_sum = 0.0
            pos_in_replay = 0
            for it in replay_items:
                rx = np.array(it.x, dtype=float)
                ry = int(it.y)
                rloss, rgw, rgb, rgnorm = model.loss_and_grad(rx, ry)
                model.step(rgw, rgb)
                replay_loss_sum += rloss
                pos_in_replay += (1 if ry == 1 else 0)
                update_item_stats(it, rloss)

                # feed per-item volatility back as "volatility" feature (for next MVI updates)
                it.meta["volatility"] = float(it.meta.get("loss_var", 0.0))

            # log
            if step % args.log_every == 0:
                m = eval_metrics(model, eval_xs, eval_ys)

                if args.policy == "memorysafe":
                    snap = buf.snapshot_stats(step)
                else:
                    # baseline snapshot for consistent plotting
                    pos = sum(1 for it in buf.items if int(it.y) == 1) if hasattr(buf, "items") else 0
                    snap = {
                        "buffer_size": len(buf),
                        "mvi_mean": 0.0,
                        "mvi_p90": 0.0,
                        "protected": 0,
                        "pos_frac": (pos / max(1, len(buf))) if len(buf) else 0.0,
                    }

                record: Dict[str, Any] = {
                    "step": step,
                    "policy": args.policy,
                    "metrics": m,
                    "buffer": snap,
                    "actions": add_info,
                    "replay": {
                        "k": len(replay_items),
                        "avg_loss": float(replay_loss_sum / max(1, len(replay_items))),
                        "pos_in_replay": int(pos_in_replay),
                    },
                }

                line = json.dumps(record, ensure_ascii=False)
                f_run.write(line + "\n")
                f_run.flush()
                f_latest.write(line + "\n")
                f_latest.flush()

                print(
                    f"[{args.policy}] step={step} "
                    f"recall_pos={m['recall_pos']:.3f} f1_pos={m['f1_pos']:.3f} "
                    f"buf={snap['buffer_size']} pos_frac={snap['pos_frac']:.2f} "
                    f"mvi_mean={snap['mvi_mean']:.3f} prot={snap['protected']} "
                    f"forgot={add_info.get('forgotten',0)} repl={add_info.get('replaced',0)}"
                )

    print(f"\nSaved logs to: {run_path}")
    print(f"Latest: runs/latest.jsonl")


if __name__ == "__main__":
    main()
