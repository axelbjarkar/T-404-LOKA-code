"""Play 5x5 Breakthrough against a trained model in the browser.

    python play/server.py path/to/forgettable_5p.pt     # then open http://localhost:8000

The model scores a state from the side to move's point of view, assuming that side moves "up" (white).
For black we rotate the board 180 degrees and swap colours, so the same model plays both sides.
The AI looks one move ahead: it picks the move after which the opponent's state scores lowest."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from loka.models import BreakthroughCNN

HERE = Path(__file__).parent
MODEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/partial_forgettable/o/models/forgettable_5p.pt")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

model = BreakthroughCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()


def moves(board, side):
    """board: 5x5 list of '.', 'w', 'b'. White moves up (row -1), black down. Yields (r, c, r2, c2)."""
    d = -1 if side == "w" else 1
    for r in range(5):
        for c in range(5):
            if board[r][c] != side:
                continue
            for dc in (-1, 0, 1):
                r2, c2 = r + d, c + dc
                if not (0 <= r2 < 5 and 0 <= c2 < 5):
                    continue
                target = board[r2][c2]
                if (dc == 0 and target == ".") or (dc != 0 and target != side):
                    yield r, c, r2, c2


def apply(board, m):
    r, c, r2, c2 = m
    b = [row[:] for row in board]
    b[r2][c2], b[r][c] = b[r][c], "."
    return b


def encode_for_mover(board, side):
    """(1, 50) tensor from the mover's view: mover = 'white' moving up."""
    b = np.array(board)
    if side == "b":
        b = b[::-1, ::-1]
        b = np.where(b == "w", "B", np.where(b == "b", "W", b))
        b = np.char.lower(b)
    return torch.FloatTensor(np.concatenate([(b == "w").ravel(), (b == "b").ravel()])[None].astype(np.float32))


def choose(board, side):
    opp = "b" if side == "w" else "w"
    goal = 0 if side == "w" else 4
    best, best_score = None, 1e9
    for m in moves(board, side):
        if m[2] == goal:
            return m  # winning move
        with torch.no_grad():
            score = model(encode_for_mover(apply(board, m), opp)).item()  # opponent's value: lower is better for us
        if score < best_score:
            best, best_score = m, score
    return best


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = (HERE / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        move = choose(req["board"], req["side"])
        body = json.dumps({"move": move}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"model {MODEL_PATH}; open http://localhost:{PORT}")
    HTTPServer(("localhost", PORT), Handler).serve_forever()
