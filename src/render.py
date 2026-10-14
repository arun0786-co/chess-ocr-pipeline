# draws fake board pictures so we have training data with known answers.
# positions come from random games so pieces sit in normal looking places.
import io
import os
import random

import chess
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = [
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
]

# (light square, dark square)
THEMES = [
    ((240, 217, 181), (181, 136, 99)),   # brown, like lichess
    ((238, 238, 210), (118, 150, 86)),   # green
    ((222, 227, 230), (140, 162, 173)),  # blue
    ((235, 235, 235), (160, 160, 160)),  # grey
    ((255, 255, 255), (200, 200, 200)),  # book
    ((255, 206, 158), (209, 139, 71)),   # orange
]

OUTLINE = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙"}
FILLED = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}


def fonts_here():
    found = [f for f in FONTS if os.path.exists(f)]
    if not found:
        raise SystemExit("no font with chess pieces found. add one to FONTS in src/render.py")
    return found


def random_position(rng):
    b = chess.Board()
    for _ in range(rng.randint(0, 90)):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            break
        b.push(rng.choice(moves))
    return b.board_fen()


def grid_from_fen(fen):
    rows = []
    for part in fen.split("/"):
        row = []
        for ch in part:
            row += ["."] * int(ch) if ch.isdigit() else [ch]
        rows.append(row)
    return rows


def draw_piece(d, x, y, sq, piece, font, style):
    white = piece.isupper()
    p = piece.upper()
    if style == "book":
        if white:
            # white body first so the dark square doesnt show through, then the black outline
            centre_text(d, x, y, sq, FILLED[p], font, (255, 255, 255), 0)
        glyph, fill, stroke = (OUTLINE[p] if white else FILLED[p]), (0, 0, 0), 0
    else:
        # screen style: filled shapes, white ones get a dark edge
        glyph, fill = FILLED[p], ((250, 250, 250) if white else (20, 20, 20))
        stroke = max(1, sq // 28)
    centre_text(d, x, y, sq, glyph, font, fill, stroke)


def centre_text(d, x, y, sq, glyph, font, fill, stroke):
    l, t, r, b = d.textbbox((0, 0), glyph, font=font, stroke_width=stroke)
    cx = x + (sq - (r - l)) / 2 - l
    cy = y + (sq - (b - t)) / 2 - t
    d.text((cx, cy), glyph, font=font, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))


def render(fen, sq=48, theme=None, font_path=None, style="book", margin=0, bg=(40, 40, 40), seed=None):
    """returns (picture, board box). board box = (left, top, right, bottom) in pixels"""
    rng = random.Random(seed)
    light, dark = theme or rng.choice(THEMES)
    font = ImageFont.truetype(font_path or rng.choice(fonts_here()), int(sq * 0.82))
    size = 8 * sq + 2 * margin
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    for r, row in enumerate(grid_from_fen(fen)):
        for c, piece in enumerate(row):
            x, y = margin + c * sq, margin + r * sq
            d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=light if (r + c) % 2 == 0 else dark)
            if piece != ".":
                draw_piece(d, x, y, sq, piece, font, style)
    return img, (margin, margin, margin + 8 * sq, margin + 8 * sq)


def roughen(img, rng):
    """make it look like a real screenshot / photo of a page: blur + jpeg"""
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(40, 95))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def random_sample(rng):
    fen = random_position(rng)
    img, box = render(fen, sq=rng.randint(24, 64), theme=rng.choice(THEMES), font_path=rng.choice(fonts_here()),
                      style=rng.choice(["book", "screen"]), margin=rng.choice([0, 0, 4, 12, 30]),
                      bg=tuple(rng.randint(20, 240) for _ in range(3)), seed=rng.random())
    return roughen(img, rng), box, fen
