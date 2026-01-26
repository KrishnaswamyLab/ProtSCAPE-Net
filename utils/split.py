"""
Data splitting utilities for ProtSCAPE datasets.
"""
import os
import random

def windowed_interpolation_split(full_dataset, window_size=10, num_windows=20, seed=0):
    rng = random.Random(seed)

    # Ensure time exists and is sortable
    for i, g in enumerate(full_dataset):
        if not hasattr(g, "time"):
            g.time = i
        try:
            g.time = int(g.time)
        except Exception:
            g.time = i

    full_dataset = sorted(full_dataset, key=lambda x: int(x.time))

    num_test_frames = window_size * num_windows
    assert num_test_frames <= len(full_dataset), "Val set exceeds available data."

    available_starts = list(range(len(full_dataset) - window_size + 1))
    rng.shuffle(available_starts)
    window_starts = available_starts[:num_windows]

    val_indices = []
    for start in window_starts:
        val_indices.extend(range(start, start + window_size))
    val_indices = sorted(set(val_indices))

    val_set = [full_dataset[i] for i in val_indices]
    train_set = [full_dataset[i] for i in range(len(full_dataset)) if i not in val_indices]

    return train_set, val_set, full_dataset

# -----------------------------
# IO utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)