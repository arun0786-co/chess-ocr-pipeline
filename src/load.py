# step 3: make a full FEN, check it with python-chess, and save it / make a lichess link
import csv
import os
from datetime import datetime

import chess


def castling(placement):
    """guess castling rights from kings and rooks still on their start squares"""
    b = chess.Board(placement + " w - - 0 1")
    rights = ""
    if b.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if b.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights += "K"
        if b.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights += "Q"
    if b.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if b.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights += "k"
        if b.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights += "q"
    return rights or "-"


def full_fen(placement, side="w"):
    return f"{placement} {side} {castling(placement)} - 0 1"


def problems(fen):
    """things that make the position impossible, in plain words"""
    b = chess.Board(fen)
    st = b.status()
    out = []
    if st & chess.STATUS_NO_WHITE_KING:
        out.append("no white king found")
    if st & chess.STATUS_NO_BLACK_KING:
        out.append("no black king found")
    if st & chess.STATUS_TOO_MANY_KINGS:
        out.append("more than one king of a colour")
    if st & chess.STATUS_PAWNS_ON_BACKRANK:
        out.append("pawn on the first or last rank")
    if st & chess.STATUS_OPPOSITE_CHECK:
        out.append("the side not to move is in check (maybe wrong side to move?)")
    if st & (chess.STATUS_TOO_MANY_WHITE_PIECES | chess.STATUS_TOO_MANY_BLACK_PIECES
             | chess.STATUS_TOO_MANY_WHITE_PAWNS | chess.STATUS_TOO_MANY_BLACK_PAWNS):
        out.append("too many pieces of one colour")
    return out


def lichess_link(fen):
    return "https://lichess.org/analysis/" + fen.replace(" ", "_")


def save_row(path, image, fen, sure):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "image", "fen", "lowest_confidence"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), image, fen, f"{sure:.2f}"])
