import random

import pytest
from PIL import Image

from src import extract, load, transform
from src.pipeline import read
from src.render import grid_from_fen, random_sample, render

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def close(a, b, px=2):
    return a is not None and all(abs(x - y) <= px for x, y in zip(a, b))


@pytest.fixture(scope="module")
def net():
    return transform.load_model()


def test_cut_squares():
    img, box = render(START, sq=30, margin=10)
    cells = extract.cut_squares(img, box)
    assert len(cells) == 64
    assert cells[0].size == (30, 30)


def test_finds_board():
    rng = random.Random(1)
    hits = sum(close(extract.find_board(img), box) for img, box, _ in (random_sample(rng) for _ in range(20)))
    assert hits >= 19


def test_finds_board_inside_screenshot():
    rng = random.Random(2)
    hits = 0
    for _ in range(15):
        img, box, _ = random_sample(rng)
        big = Image.new("RGB", (img.width + 250, img.height + 140), (245, 245, 245))
        big.paste((50, 50, 60), (0, 0, big.width, 45))  # a menu bar
        big.paste(img, (30, 70))
        hits += close(extract.find_board(big), (box[0] + 30, box[1] + 70, box[2] + 30, box[3] + 70))
    assert hits >= 13


def test_no_board_in_plain_picture():
    assert extract.find_board(Image.new("RGB", (300, 300), (200, 200, 200))) is None


def test_reads_boards(net):
    rng = random.Random(3)
    right = 0
    for _ in range(20):
        img, _, fen = random_sample(rng)
        right += read(img, model=net, flipped=False)["fen"].split()[0] == fen
    assert right >= 18


def test_start_position_full_fen(net):
    img, _ = render(START, sq=40, style="screen")
    res = read(img, model=net)
    assert res["fen"] == START + " w KQkq - 0 1"
    assert res["problems"] == []


def test_flipped_board(net):
    img, _ = render(START[::-1], sq=40)  # black at the bottom
    res = read(img, model=net)
    assert res["flipped"]
    assert res["fen"].split()[0] == START


def test_placement_and_flip():
    labels = [c for row in grid_from_fen(START) for c in row]
    assert transform.to_placement(labels) == START
    assert not transform.looks_flipped(labels)
    assert transform.looks_flipped(transform.flip(labels))


def test_fen_tools():
    assert load.castling("r3k2r/8/8/8/8/8/8/4K2R") == "Kkq"
    assert load.castling("8/8/8/8/8/8/8/8") == "-"
    assert "no white king found" in load.problems("4k3/8/8/8/8/8/8/8 w - - 0 1")
    assert "pawn on the first or last rank" in load.problems("P3k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert load.lichess_link("8/8 w - - 0 1") == "https://lichess.org/analysis/8/8_w_-_-_0_1"
