# step 1: find the board in a picture and cut it into 64 squares.
# works on screenshots and book diagrams. not on photos taken at an angle.
import numpy as np
from PIL import Image


def load(path):
    return Image.open(path).convert("RGB")


def gray(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def edge_profile(g, axis):
    """how strong the edges are at each pixel boundary. index i = edge just before pixel i"""
    d = np.abs(np.diff(g, axis=1 if axis == "x" else 0)).sum(axis=0 if axis == "x" else 1)
    prof = np.zeros(len(d) + 2, dtype=np.float32)
    prof[1:-1] = d
    # the picture border can be the board edge (when the board is cropped tight)
    prof[0] = prof[-1] = 0.5 * d.max()
    # a line can be 1 px off, so allow a little wiggle
    return np.maximum(np.maximum(prof, np.roll(prof, 1)), np.roll(prof, -1))


def grid_candidates(prof, top=4, min_sq=12, sizes=None):
    """best (start, square size) pairs where 9 evenly spaced lines sit on strong edges"""
    n = len(prof) - 1
    found = []
    for s in sizes or range(min_sq, n // 8 + 1):
        starts = np.arange(0, n - 8 * s + 1)
        if not len(starts):
            continue
        idx = starts[:, None] + s * np.arange(9)[None, :]
        scores = prof[idx].mean(axis=1)
        # keep a few starts per size, other edges (menus, borders) can beat the real one.
        # they must be half a square apart, else 3 neighbours of one edge take all the places
        picked = []
        for i in np.argsort(scores)[::-1]:
            if all(abs(int(starts[i]) - o) > s // 2 for o in picked):
                picked.append(int(starts[i]))
                found.append((float(scores[i]), int(starts[i]), s))
                if len(picked) == 3:
                    break
    found.sort(reverse=True)
    out = []
    for sc, o, s in found:
        if all(abs(o - o2) > s2 // 2 or abs(s - s2) > 1 for _, o2, s2 in out):
            out.append((sc, o, s))
        if len(out) == top:
            break
    return out


def checker_score(g, x0, y0, sx, sy):
    """a real board has light and dark squares taking turns in EVERY row and column.
    medians ignore the pieces. a grid one row off has a row of background -> score near 0"""
    m = np.empty((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            cell = g[y0 + r * sy: y0 + (r + 1) * sy, x0 + c * sx: x0 + (c + 1) * sx]
            m[r, c] = np.median(cell[::2, ::2])  # every 2nd pixel is enough, and faster
    par = (np.add.outer(np.arange(8), np.arange(8)) % 2).astype(bool)
    sign = 1.0 if m[~par].mean() >= m[par].mean() else -1.0
    rows = [sign * (m[i][~par[i]].mean() - m[i][par[i]].mean()) for i in range(8)]
    cols = [sign * (m[:, i][~par[:, i]].mean() - m[:, i][par[:, i]].mean()) for i in range(8)]
    spread = m[~par].std() + m[par].std()
    return min(rows + cols) / (spread + 1.0)


def near(s):
    d = max(2, round(0.06 * s))
    return range(max(12, s - d), s + d + 1)


def find_board(img):
    """returns (left, top, right, bottom) of the board, or None"""
    g = gray(img)
    h, w = g.shape
    tried = set()
    best, box = 0.0, None
    for _, x0, sx in grid_candidates(edge_profile(g, "x"), top=6):
        # rows: only under these columns (so menus next to the board dont count),
        # and only square sizes close to sx because squares are square
        cols = g[:, x0:x0 + 8 * sx]
        for _, y0, sy in grid_candidates(edge_profile(cols, "y"), top=6, sizes=near(sx)):
            # and check the columns again using only these rows
            rows = g[y0:y0 + 8 * sy, :]
            for _, x1, sx1 in grid_candidates(edge_profile(rows, "x"), top=3, sizes=near(sy)):
                # a border next to the board can look like a board line, so also try one square over
                for dx in (-sx1, 0, sx1):
                    for dy in (-sy, 0, sy):
                        x, y = x1 + dx, y0 + dy
                        if x < 0 or y < 0 or x + 8 * sx1 > w or y + 8 * sy > h or (x, y, sx1, sy) in tried:
                            continue
                        tried.add((x, y, sx1, sy))
                        sc = checker_score(g, x, y, sx1, sy)
                        if sc > best:
                            best, box = sc, (x, y, x + 8 * sx1, y + 8 * sy)
    return box if best > 0.8 else None


def cut_squares(img, box=None):
    """64 squares, row by row from the top left (a8 ... h1 when white is at the bottom)"""
    if box is None:
        box = (0, 0, img.width, img.height)
    l, t, r, b = box
    sx, sy = (r - l) / 8, (b - t) / 8
    out = []
    for row in range(8):
        for col in range(8):
            out.append(img.crop((round(l + col * sx), round(t + row * sy),
                                 round(l + (col + 1) * sx), round(t + (row + 1) * sy))))
    return out
