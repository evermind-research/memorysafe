# baselines.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class MemoryItem:
    x: List[float]
    y: int
    step: int
    meta: Dict[str, Any]


class BaseBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.items: List[MemoryItem] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: MemoryItem) -> Dict[str, Any]:
        raise NotImplementedError

    def sample(self, k: int) -> List[MemoryItem]:
        if len(self.items) == 0 or k <= 0:
            return []
        k = min(k, len(self.items))
        return self.rng.sample(self.items, k)


class FIFOBuffer(BaseBuffer):
    def add(self, item: MemoryItem) -> Dict[str, Any]:
        self.seen += 1
        evicted = 0
        if len(self.items) >= self.capacity:
            self.items.pop(0)
            evicted = 1
        self.items.append(item)
        return {"added": 1, "evicted": evicted, "policy": "fifo"}


class ReservoirBuffer(BaseBuffer):
    def add(self, item: MemoryItem) -> Dict[str, Any]:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return {"added": 1, "evicted": 0, "policy": "reservoir"}
        # classic reservoir
        j = self.rng.randint(0, self.seen - 1)
        if j < self.capacity:
            self.items[j] = item
            return {"added": 1, "evicted": 1, "policy": "reservoir"}
        return {"added": 0, "evicted": 0, "policy": "reservoir"}


class RandomReplayBuffer(FIFOBuffer):
    """
    Same storage as FIFO, but used as a named baseline.
    Replay sampling is uniform (handled by BaseBuffer.sample).
    """
    pass


def make_baseline(policy: str, capacity: int, seed: int) -> BaseBuffer:
    policy = policy.lower().strip()
    if policy == "fifo":
        return FIFOBuffer(capacity, seed)
    if policy == "reservoir":
        return ReservoirBuffer(capacity, seed)
    if policy in ("random", "random_replay"):
        return RandomReplayBuffer(capacity, seed)
    raise ValueError(f"Unknown baseline policy: {policy}")
