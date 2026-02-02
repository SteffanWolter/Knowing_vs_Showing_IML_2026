from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


@dataclass(frozen=True)
class ProbeTrainingResult:
    probe: LogisticRegression
    cv_mean: float
    cv_std: float


def train_logreg_probe(
    *,
    states: Iterable[torch.Tensor],
    labels: Iterable[int],
    seed: int,
    cv_folds: int = 5,
) -> ProbeTrainingResult:
    xs = torch.stack(list(states)).to(torch.float32).cpu().numpy()
    ys = np.array(list(labels), dtype=np.int64)

    clf = LogisticRegression(random_state=seed, max_iter=1000)
    scores = cross_val_score(clf, xs, ys, cv=cv_folds)
    clf.fit(xs, ys)
    return ProbeTrainingResult(probe=clf, cv_mean=float(scores.mean()), cv_std=float(scores.std()))
