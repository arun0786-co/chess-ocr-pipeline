# chess ocr pipeline

give it a picture of a chess board (screenshot, book diagram), get back the FEN and a lichess link to analyse it.

```
$ python -m src.pipeline read data/samples/board_3.png
rnbqk2r/1ppp1ppp/p3p3/4B1N1/4nPP1/1P6/P1PPPb1P/RN1QKB1R w KQkq - 0 1
https://lichess.org/analysis/rnbqk2r/1ppp1ppp/...
```

## how it works

its a small pipeline, 3 steps:

1. **extract** (`src/extract.py`) - find the board in the picture and cut it into 64 squares. the lines between squares make strong edges that go across the whole board, so i look for 9 lines with the same gap. then i check that the squares really go light / dark / light in every row and column.
2. **transform** (`src/transform.py`) - for every square, take away the square colour and shrink it to 24x24. a small neural net (1 hidden layer, written with just numpy) says which piece it is. 64 answers -> FEN.
3. **load** (`src/load.py`) - make the full FEN (castling is guessed from kings and rooks), check it with python-chess, make the lichess link, and save to a csv if you want.

there was no dataset, so `src/render.py` draws fake boards: positions from random games, 6 colour themes, 2 piece styles, different sizes, blur and jpeg noise. the net is trained only on these.

## results

on 150 new fake boards it has never seen:

| test | fully correct boards |
| --- | --- |
| just the board | 148 / 150 |
| board inside a bigger "screenshot" with a menu bar | 145 / 150 |
| single squares | 99.95% |

the model is small (~450 KB) and training takes about 40 sec on a laptop.

## use it

```
pip install -r requirements.txt
python -m src.pipeline read board.png
python -m src.pipeline read board.png --side b        # black to move
python -m src.pipeline read board.png --save games.csv
python -m src.pipeline train                          # make new fake boards and train again
python -m src.pipeline samples                        # save some sample pictures
```

if black is at the bottom of the picture it turns the board around by itself (it looks where the kings and pawns are). use `--flip` to force it.

there is also `notebooks/walkthrough.ipynb` that shows every step.

## limits

- works on flat, straight pictures (screenshots, scans). not on phone photos of a real board at an angle.
- trained on unicode chess fonts. the pieces on chess.com / lichess look a bit different, so it can make mistakes there. training on some real screenshots would fix it (todo).
- side to move, en passant and move numbers can't be seen in a picture, so they are guessed / default.

## tests

```
pytest
```

## todo

- train with real screenshots from lichess / chess.com piece sets
- photos of real boards (needs perspective fix)
- read a whole book page with many diagrams
