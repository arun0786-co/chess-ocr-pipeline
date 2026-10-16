# step 2: look at each square and say which piece is on it -> FEN.
# small neural net written with numpy only (1 hidden layer), so no big ML library is needed.
import os

import numpy as np

LABELS = list(".PNBRQKpnbrqk")  # . = empty square
SIZE = 24  # squares get shrunk to 24x24
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "model.npz")


def features(cell):
    """signed 'ink' compared to the square colour, so board colours dont matter"""
    g = np.asarray(cell.convert("L").resize((SIZE, SIZE)), dtype=np.float32)
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = np.median(ring)
    return ((g - bg) / 128.0).ravel()


def board_features(cells):
    return np.stack([features(c) for c in cells])


# ---- the net ----

def init(n_in, n_hidden, n_out, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "w1": (rng.standard_normal((n_in, n_hidden)) * np.sqrt(2 / n_in)).astype(np.float32),
        "b1": np.zeros(n_hidden, np.float32),
        "w2": (rng.standard_normal((n_hidden, n_out)) * np.sqrt(1 / n_hidden)).astype(np.float32),
        "b2": np.zeros(n_out, np.float32),
    }


def forward(net, x):
    h = np.maximum(0, x @ net["w1"] + net["b1"])
    z = h @ net["w2"] + net["b2"]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return h, p / p.sum(axis=1, keepdims=True)


def train(x, y, hidden=192, epochs=20, lr=2e-3, batch=256, seed=0, log=print):
    """plain adam + softmax cross entropy"""
    net = init(x.shape[1], hidden, len(LABELS), seed)
    m = {k: np.zeros_like(v) for k, v in net.items()}
    v = {k: np.zeros_like(w) for k, w in net.items()}
    rng = np.random.default_rng(seed)
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(x))
        loss = 0.0
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            xb, yb = x[idx], y[idx]
            h, p = forward(net, xb)
            loss += -np.log(p[np.arange(len(yb)), yb] + 1e-9).sum()
            d2 = p.copy()
            d2[np.arange(len(yb)), yb] -= 1
            d2 /= len(yb)
            dh = (d2 @ net["w2"].T) * (h > 0)
            grads = {"w2": h.T @ d2, "b2": d2.sum(0), "w1": xb.T @ dh, "b1": dh.sum(0)}
            step += 1
            for k in net:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                mh = m[k] / (1 - 0.9 ** step)
                vh = v[k] / (1 - 0.999 ** step)
                net[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if log:
            log(f"epoch {ep + 1}: loss {loss / len(x):.4f}")
    return net


def save(net, path=MODEL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **net)


def load_model(path=MODEL):
    if not os.path.exists(path):
        raise SystemExit("no model yet. run:  python -m src.pipeline train")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def predict(net, x):
    """labels and how sure the net is for each square"""
    _, p = forward(net, x.astype(np.float32))
    return [LABELS[i] for i in p.argmax(1)], p.max(1)


# ---- board -> fen ----

def to_placement(labels):
    rows = []
    for r in range(8):
        row, empty = "", 0
        for c in labels[r * 8:(r + 1) * 8]:
            if c == ".":
                empty += 1
            else:
                row += (str(empty) if empty else "") + c
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)


def looks_flipped(labels):
    """black at the bottom? check where the kings and pawns are"""
    top = labels[:32]
    bottom = labels[32:]
    white_low = bottom.count("K") + bottom.count("P") - top.count("K") - top.count("P")
    black_low = bottom.count("k") + bottom.count("p") - top.count("k") - top.count("p")
    return black_low > white_low


def flip(labels):
    return labels[::-1]
