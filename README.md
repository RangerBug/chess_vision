# Chess Vision

A **deep learning–based computer vision system** that tracks live, over-the-board chess games from video. The program detects board state changes and logs moves in **PGN format**, enabling downstream analysis with standard chess engines.

---

## How It Works

1. **Board Calibration**
   The user selects the four board corners (a8, h8, h1, a1). The board is reprojected into a top-down view.

2. **Square Classification**
   The reprojected board is split into **64 individual squares**. Each square is passed to a deep learning model that classifies it as **occupied or empty**.

   * Assumes a standard starting position
   * Assumes White plays the first move

3. **Move Detection**
   During gameplay, the system waits for a still frame, processes it, and compares board occupancy:

   * **Occupied → Empty** squares → candidate start squares
   * **Empty → Occupied** squares → candidate end squares

   Legal move inference is used to determine the most likely move played.

   ⚠️ Fast captures may be missed if the occupancy change is not observed in time (high-priority improvement).

---

## Datasets

* **Initial dataset:** Self-created, included in this repository
  ~1,000 square images (50% empty, 50% occupied)

* **Synthetic dataset (current):**
  [https://www.kaggle.com/datasets/thefamousrat/synthetic-chess-board-images](https://www.kaggle.com/datasets/thefamousrat/synthetic-chess-board-images)

---

## Usage
uv run python src/chess_vision/main.py
* Start the video stream
* **`c`** – Calibrate board (click a8, h8, h1, a1 in order)
* **`p`** – Pause
* **`e`** – End game, save PGN, exit
* **`q`** – Quit without saving

Recalibration can be done at any time, but the game log resets only on program restart.

---

## Installation
(Please lmk if more is needed I'm just guessing this works for now)

```bash
git clone https://github.com/RangerBug/chess_vision.git
cd chess_vision
pip install uv
uv sync
uv run python src/chess_vision/main.py
```

---

## TODO / WIP

* Automatic board corner tracking
* Full piece classification (not just occupancy)
* Faster frame processing
* Integrate a chess engine for real-time analysis
* Video overlays (moves, engine eval, timing, UI)