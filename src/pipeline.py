# picture of a chess board -> FEN.  extract -> transform -> load
#
#   python -m src.pipeline read board.png            read one picture
#   python -m src.pipeline read board.png --side b   black to move
#   python -m src.pipeline train                     make fake boards and train the model
#   python -m src.pipeline samples                   save a few sample pictures in data/samples
import argparse
import os
import random
import sys

import numpy as np

from . import extract, load, transform
from .render import grid_from_fen, random_sample

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def read(path, side="w", flipped=None, model=None):
    img = extract.load(path) if isinstance(path, str) else path
    box = extract.find_board(img)
    if box is None:
        box = (0, 0, img.width, img.height)  # maybe its already cropped
        found = False
    else:
        found = True
    cells = extract.cut_squares(img, box)
    net = model or transform.load_model()
    labels, sure = transform.predict(net, transform.board_features(cells))
    if flipped is None:
        flipped = transform.looks_flipped(labels)
    if flipped:
        labels, sure = transform.flip(labels), sure[::-1]
    fen = load.full_fen(transform.to_placement(labels), side)
    return {"fen": fen, "box": box, "found": found, "flipped": flipped,
            "lowest": float(sure.min()), "unsure": [i for i, s in enumerate(sure) if s < 0.8],
            "problems": load.problems(fen)}


def make_set(n, seed, jitter=2):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        img, box, fen = random_sample(rng)
        # the finder can be a pixel or two off, so train on slightly moved boxes too
        l, t, r, b = (v + rng.randint(-jitter, jitter) for v in box)
        cells = extract.cut_squares(img, (l, t, r, b))
        xs.append(transform.board_features(cells))
        ys += [transform.LABELS.index(ch) for row in grid_from_fen(fen) for ch in row]
    return np.concatenate(xs), np.array(ys)


def accuracy(net, x, y):
    pred, _ = transform.predict(net, x)
    return float(np.mean([transform.LABELS.index(p) == t for p, t in zip(pred, y)]))


def cmd_train(a):
    print(f"making {a.boards} training boards...")
    x, y = make_set(a.boards, seed=1)
    print(f"making {a.boards // 5} test boards...")
    xt, yt = make_set(a.boards // 5, seed=999)
    net = transform.train(x, y, epochs=a.epochs)
    print(f"square accuracy: train {accuracy(net, x, y):.4f}, test {accuracy(net, xt, yt):.4f}")
    transform.save(net)
    print("saved models/model.npz")


def cmd_read(a):
    res = read(a.image, side=a.side, flipped=True if a.flip else None)
    if not res["found"]:
        print("could not find the board, using the whole picture. crop it to the board for better results.")
    print(res["fen"])
    print(load.lichess_link(res["fen"]))
    if res["flipped"]:
        print("(looks like black is at the bottom, turned it around. use --flip to force it)")
    if res["unsure"]:
        names = [f"{'abcdefgh'[i % 8]}{8 - i // 8}" for i in res["unsure"]]
        print(f"not so sure about: {', '.join(names)}")
    for p in res["problems"]:
        print("warning:", p)
    if a.save:
        load.save_row(a.save, a.image, res["fen"], res["lowest"])
        print("saved to", a.save)


def cmd_samples(a):
    out = os.path.join(DATA, "samples")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(a.seed)
    with open(os.path.join(out, "answers.txt"), "w") as f:
        for i in range(a.n):
            img, _, fen = random_sample(rng)
            name = f"board_{i + 1}.png"
            img.save(os.path.join(out, name))
            f.write(f"{name} {fen}\n")
    print(f"saved {a.n} pictures in data/samples")


def main(argv=None):
    p = argparse.ArgumentParser(prog="chess-ocr")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="read a board picture")
    r.add_argument("image")
    r.add_argument("--side", default="w", choices=["w", "b"], help="who is to move")
    r.add_argument("--flip", action="store_true", help="black is at the bottom")
    r.add_argument("--save", help="add the result to this csv file")
    t = sub.add_parser("train", help="train the square model on fake boards")
    t.add_argument("--boards", type=int, default=3000)
    t.add_argument("--epochs", type=int, default=25)
    s = sub.add_parser("samples", help="save sample pictures")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    {"read": cmd_read, "train": cmd_train, "samples": cmd_samples}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
