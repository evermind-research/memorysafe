# memorysafe_buffer.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from baselines import MemoryItem, BaseBuffer


@dataclass
class GovernanceStats:
    protected: int = 0
    forgotten: int = 0
    replaced: int = 0
    added: int = 0


class RunningNorm:
    """Welford running mean/var for normalization."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 1e-8

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-8))

    def z(self, x: float) -> float:
        return (x - self.mean) / (self.std + 1e-8)


class MemorySafeBuffer(BaseBuffer):
    """
    Memory governance demo:
      - computes MVI per item (vulnerability proxy)
      - "protects" high-risk items
      - "forgets" low-value / noisy / volatile items
      - replay prioritizes high MVI (what's about to be forgotten)
    """

    def __init__(
        self,
        capacity: int,
        seed: int = 0,
        mvi_forget_th: float = 0.8,
        min_age_forget: int = 200,
        protect_top_frac: float = 0.15,
        forget_frac_when_full: float = 0.05,
    ):
        super().__init__(capacity, seed)
        self.mvi_forget_th = float(mvi_forget_th)
        self.min_age_forget = int(min_age_forget)
        self.protect_top_frac = float(protect_top_frac)
        self.forget_frac_when_full = float(forget_frac_when_full)

        # normalizers
        self.loss_norm = RunningNorm()
        self.grad_norm = RunningNorm()
        self.novelty_norm = RunningNorm()
        self.vol_norm = RunningNorm()

        self.gov = GovernanceStats()

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _compute_mvi(self, loss: float, grad: float, novelty: float, volatility: float) -> float:
        # Normalize to z-scores, then squash to 0..1
        z_loss = self.loss_norm.z(loss)
        z_grad = self.grad_norm.z(grad)
        z_nov = self.novelty_norm.z(novelty)
        z_vol = self.vol_norm.z(volatility)

        # Weighted mix: vulnerability rises with loss/grad/novelty/volatility
        raw = 0.45 * z_loss + 0.30 * z_grad + 0.15 * z_nov + 0.10 * z_vol
        return self._sigmoid(raw)

    def _item_age(self, it: MemoryItem, now_step: int) -> int:
        return max(0, now_step - it.step)

    def _value_score(self, it: MemoryItem) -> float:
        """
        Value proxy: protect rare positives + items that historically helped.
        meta['utility_ema'] is updated during replay usage by trainer.
        """
        y = int(it.y)
        util = float(it.meta.get("utility_ema", 0.0))
        rare_boost = 0.5 if y == 1 else 0.0
        return util + rare_boost

    def _volatility(self, it: MemoryItem) -> float:
        """
        Volatility proxy: how unstable loss is on this item across replays.
        meta['loss_var'] provided by trainer (EMA variance style).
        """
        return float(it.meta.get("loss_var", 0.0))

    def _refresh_protection_flags(self, now_step: int) -> int:
        """Mark top fraction as protected (by mvi)."""
        if not self.items:
            return 0
        sorted_idx = sorted(range(len(self.items)), key=lambda i: float(self.items[i].meta.get("mvi", 0.0)), reverse=True)
        k = max(1, int(self.protect_top_frac * len(self.items)))
        protected_set = set(sorted_idx[:k])
        count = 0
        for i, it in enumerate(self.items):
            prot = i in protected_set
            it.meta["protected"] = prot
            if prot:
                count += 1
        return count

    def add(self, item: MemoryItem) -> Dict[str, Any]:
        """
        Expects item.meta to include:
          - loss, grad_norm, novelty, volatility (optional; defaults to 0)
        """
        self.seen += 1

        loss = float(item.meta.get("loss", 0.0))
        grad = float(item.meta.get("grad_norm", 0.0))
        novelty = float(item.meta.get("novelty", 0.0))
        volatility = float(item.meta.get("volatility", 0.0))

        # update running stats first
        self.loss_norm.update(loss)
        self.grad_norm.update(grad)
        self.novelty_norm.update(novelty)
        self.vol_norm.update(volatility)

        mvi = self._compute_mvi(loss, grad, novelty, volatility)
        item.meta["mvi"] = mvi
        item.meta.setdefault("protected", False)
        item.meta.setdefault("utility_ema", 0.0)
        item.meta.setdefault("loss_var", 0.0)

        action = {"policy": "memorysafe", "added": 0, "evicted": 0, "forgotten": 0, "protected": 0, "replaced": 0}

        # If buffer not full, just add
        if len(self.items) < self.capacity:
            self.items.append(item)
            self.gov.added += 1
            action["added"] = 1
            # refresh protection
            prot = self._refresh_protection_flags(item.step)
            action["protected"] = prot
            return action

        # If full, run forgetting pass (small) then decide insertion
        forgot = self._forget_pass(now_step=item.step)
        action["forgotten"] = forgot

        # If space created, add
        if len(self.items) < self.capacity:
            self.items.append(item)
            self.gov.added += 1
            action["added"] = 1
            prot = self._refresh_protection_flags(item.step)
            action["protected"] = prot
            return action

        # Otherwise replace weakest candidate (low value, low mvi, high volatility)
        idx = self._choose_replacement(now_step=item.step)
        if idx is not None:
            self.items[idx] = item
            self.gov.replaced += 1
            action["replaced"] = 1
            action["evicted"] = 1
            prot = self._refresh_protection_flags(item.step)
            action["protected"] = prot
        return action

    def _forget_pass(self, now_step: int) -> int:
        if not self.items:
            return 0
        n_forget = max(1, int(self.forget_frac_when_full * len(self.items)))
        # rank by "forget score": low value + high volatility + not protected + old enough
        candidates = []
        for i, it in enumerate(self.items):
            age = self._item_age(it, now_step)
            if age < self.min_age_forget:
                continue
            if bool(it.meta.get("protected", False)):
                continue

            mvi = float(it.meta.get("mvi", 0.0))
            val = self._value_score(it)
            vol = self._volatility(it)

            # Forget if: either explicitly risky-noisy, or generally low value
            # score higher => more likely to be forgotten
            forget_score = (1.0 - val) + 0.6 * vol + 0.2 * (1.0 - mvi)
            # extra push if it's above threshold and low value: "fragile & not valuable"
            if mvi > self.mvi_forget_th and val < 0.2:
                forget_score += 0.8

            candidates.append((forget_score, i))

        if not candidates:
            return 0

        candidates.sort(reverse=True)
        to_remove = [i for _, i in candidates[:n_forget]]
        to_remove_set = set(to_remove)

        new_items = []
        forgotten = 0
        for i, it in enumerate(self.items):
            if i in to_remove_set:
                forgotten += 1
            else:
                new_items.append(it)
        self.items = new_items
        self.gov.forgotten += forgotten
        return forgotten

    def _choose_replacement(self, now_step: int) -> Optional[int]:
        # choose a non-protected item with minimal "keep score"
        best_i = None
        best_score = float("inf")
        for i, it in enumerate(self.items):
            if bool(it.meta.get("protected", False)):
                continue
            mvi = float(it.meta.get("mvi", 0.0))
            val = self._value_score(it)
            vol = self._volatility(it)
            age = self._item_age(it, now_step)

            # lower keep_score => easier to replace
            keep_score = 1.2 * val + 0.6 * mvi - 0.5 * vol + 0.0005 * age
            if keep_score < best_score:
                best_score = keep_score
                best_i = i
        return best_i

    def sample(self, k: int) -> List[MemoryItem]:
        """
        Prioritized replay: pick high MVI items more often.
        """
        if len(self.items) == 0 or k <= 0:
            return []
        k = min(k, len(self.items))

        # Build weights from MVI (ensure >0)
        weights = []
        for it in self.items:
            mvi = float(it.meta.get("mvi", 0.0))
            w = 0.2 + 0.8 * mvi
            weights.append(w)

        # weighted sample without replacement
        chosen: List[MemoryItem] = []
        idxs = list(range(len(self.items)))
        for _ in range(k):
            total = sum(weights[i] for i in idxs)
            r = self.rng.random() * total
            acc = 0.0
            pick = idxs[-1]
            for i in idxs:
                acc += weights[i]
                if acc >= r:
                    pick = i
                    break
            chosen.append(self.items[pick])
            idxs.remove(pick)
        return chosen

    def snapshot_stats(self, now_step: int) -> Dict[str, Any]:
        if not self.items:
            return {
                "buffer_size": 0,
                "mvi_mean": 0.0,
                "mvi_p90": 0.0,
                "protected": 0,
                "pos_frac": 0.0,
            }
        mvis = sorted(float(it.meta.get("mvi", 0.0)) for it in self.items)
        mvi_mean = sum(mvis) / len(mvis)
        p90 = mvis[int(0.9 * (len(mvis) - 1))]
        prot = sum(1 for it in self.items if bool(it.meta.get("protected", False)))
        pos = sum(1 for it in self.items if int(it.y) == 1)
        return {
            "buffer_size": len(self.items),
            "mvi_mean": mvi_mean,
            "mvi_p90": p90,
            "protected": prot,
            "pos_frac": pos / len(self.items),
        }
