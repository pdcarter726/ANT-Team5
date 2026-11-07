# Hands-free arcade

Six browser games played with the eego cap + the webcam eye cursor (`gaze/`). Plain HTML/JS: no install,
no build step, nothing to download (works offline).

| Game | EEG + Gaze controls | Eyes only |
|---|---|---|
| Flappy Bird | jaw clench or eyebrow raise = flap, double blink = pause | bird flies toward where you look |
| Simon | look + clench a color (clench hold = hold the color down), double blink = replay, eyebrow raise = quick actions. Also "Simon Says" with every gesture | look and hold on a color |
| Chess | look + clench a piece, then a square; or clench hold on a piece, look at the square, relax (drag). Double blink = deselect | look and hold on a piece, then on a square |
| Checkers | same as chess; multi-jumps one hop at a time | same as chess |
| Tic Tac Toe | look + clench a square | look and hold on a square |
| Air Hockey | your paddle follows the cursor (head turns / eyes) | same |

Everywhere: **eyebrow raise** (Space) opens the menu (except Flappy, where it flaps); **Esc** opens it for the
operator. Menus and buttons can also be picked by holding the gaze on them. The **Input** button in the top
bar switches between *EEG + Gaze* and *Eyes only* (ignores all gestures, selects by look-and-hold) and is
remembered for every game.

## Run
**From the controller dashboard (normal way):** the games play in the **Games** tab.
```
.venv\Scripts\python.exe -m controller --model ml/model.py:predict --gaze-profile <name> --arcade
```
`--arcade` opens that tab at start and uses the `arcade` mapping (`--arcade chess` opens one game).
Pick a game from the strip at the top of the tab. Without `--model` the dashboard runs in rehearsal
mode (simulated EEG, test buttons in "Mapping and test"). Needs `PySide6-WebEngine`.

**Games only (no EEG):** `.venv\Scripts\python.exe games\serve.py` (or double-click `arcade.bat`;
`python3 games/serve.py` on Linux/macOS) serves `http://127.0.0.1:8765/` and opens the browser. Stop with
Ctrl+C. Options: `--game chess`, `--port 9000`, `--no-browser`. If this is already running, the dashboard
reuses it.

## How gestures reach the games
The browser only sees normal input, produced by the controller with `controller/mappings/arcade.json`:

| Gesture | Controller sends | Games read |
|---|---|---|
| Head turn up/down/left/right | cursor movement | aim / paddle |
| Jaw clench (short) | left click | press / select |
| Jaw clench hold (> 0.5 s) | left button held until the clench ends | Simon: color held; chess/checkers: drag |
| Double blink | right click | replay / deselect / pause |
| Eyebrow raise | Space | menu (Flappy: flap) |
| Winks | nothing | - |

Inside the page a left button held longer than 0.3 s counts as a clench hold (the controller only presses
the button down once the clench has lasted its own `hold_s`).

## Files
- `index.html` arcade menu + shared settings (look-and-hold times, sound, input log)
- `shared/hf.js` input -> gestures, look-and-hold, menus, sounds, best scores; `shared/hf.css` shared look
- `flappy/`, `simon/`, `chess/`, `checkers/`, `tictactoe/`, `airhockey/` one folder per game (`simon/` has its own README)
- `serve.py` local web server
