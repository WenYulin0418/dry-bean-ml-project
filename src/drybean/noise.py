import numpy as np


def add_feature_noise(
    x,
    noise_type: str,
    strength: float,
    seed: int,
) -> np.ndarray:
    if not 0 <= strength <= 1:
        raise ValueError("噪声强度必须位于 [0, 1]")
    result = np.asarray(x, dtype=float).copy()
    if strength == 0:
        return result
    rng = np.random.default_rng(seed)
    if noise_type == "gaussian":
        scale = np.nanstd(result, axis=0, keepdims=True)
        result += rng.normal(0.0, strength, size=result.shape) * scale
    elif noise_type == "impulse":
        mask = rng.random(result.shape) < strength
        low = np.nanquantile(result, 0.001, axis=0)
        high = np.nanquantile(result, 0.999, axis=0)
        low_grid = np.broadcast_to(low, result.shape)
        high_grid = np.broadcast_to(high, result.shape)
        choose_high = rng.random(result.shape) >= 0.5
        result[mask] = np.where(
            choose_high[mask], high_grid[mask], low_grid[mask]
        )
    elif noise_type == "missing":
        result[rng.random(result.shape) < strength] = np.nan
    else:
        raise ValueError(f"未知特征噪声: {noise_type}")
    return result


def flip_labels(
    y,
    strength: float,
    n_classes: int,
    seed: int,
) -> np.ndarray:
    if not 0 <= strength <= 1:
        raise ValueError("噪声强度必须位于 [0, 1]")
    if n_classes < 2:
        raise ValueError("类别数必须至少为 2")
    result = np.asarray(y).copy()
    if strength == 0:
        return result
    rng = np.random.default_rng(seed)
    mask = rng.random(len(result)) < strength
    offsets = rng.integers(1, n_classes, size=int(mask.sum()))
    result[mask] = (result[mask] + offsets) % n_classes
    return result

