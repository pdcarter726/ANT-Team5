# Simon (hands-free browser game)

A Simon memory game and a "Simon Says" gesture game, played with the eego cap + the webcam eye cursor
(`gaze/`). Plain HTML/JS, no install, no build step.

## Run
Part of the arcade: `python games/serve.py --game simon` (see `games/README.md`). Add `#eeg`, `#gaze` or
`#says` to the URL to jump straight into a mode. Run the controller with `--mapping arcade`.

Clench hold: in Classic Simon a held clench keeps the color lit and sounding until you relax (the press
counts as soon as the clench starts). Make one click on the page first so the
browser has focus and allows sound.

## Bindings
The game only reads normal mouse/keyboard input, so the EEG app must emit exactly this (e.g. with pyautogui):

| Gesture | Input the EEG app sends | In the game |
|---|---|---|
| Head turn / eyes | cursor movement (gaze app) | aim at a color; head turns in Simon Says |
| Jaw clench | left click | press the color under the cursor / select a button |
| Clench hold | left button held >= 0.3 s | Simon Says gesture |
| Double blink | right click | replay the sequence |
| Eyebrow raise | Space | quick actions menu (resume, replay, restart, main menu) |
| Left / right wink | not linked | - |

Bindings live in `bindings.js` (hold threshold, which key is the eyebrow raise, labels).
Tip: call `gaze.control.freeze(0.8)` from the EEG app when a gesture starts so the cursor doesn't
slide off the color while clenching.

## Modes
- **Classic Simon (EEG + gaze)**: watch the sequence, then look at each color and clench your jaw to press it.
  Look at the center circle and clench to start.
- **Look & Hold (eyes only)**: the same game using only the eye cursor. Hold your gaze on a color until its
  ring fills (1.0 s by default) to press it; hold on the center to start, and mid-game hold on the center
  for quick actions. Clicks and keys are ignored (logged as "ignored"). Short slips off a color (under
  0.25 s) pause the ring instead of resetting it, because webcam gaze jitters by 150-350 px.
- **Simon Says (EEG gestures)**: a gesture is shown; do it only if it starts with "Simon says", otherwise
  stay still until the timer runs out. 3 lives, the time window shrinks each round. Head turns count when
  the cursor moves more than 25% of the screen from where it was when the cue appeared.
  Every detected gesture lights up its chip at the bottom, so this mode doubles as a live classifier check.

Menus and the quick-actions tiles can also be selected by holding the gaze on them (1.5 s).

## Settings (operator)
Settings button on the main menu: look-and-hold times, sequence speed, head-turn distance, head turns
on/off in Simon Says, sound, and the detected-input log (bottom left). Saved in the browser.

Test keys without the cap: `1`-`4` press a color (Classic), `Space` eyebrow raise, `J` jaw clench,
`H` clench hold, `B` double blink, arrow keys head turns, `Esc` quick actions. A real mouse also works.

## Files
- `index.html` screens and overlays
- `style.css` layout (large targets for webcam gaze)
- `bindings.js` gesture-to-input bindings
- `game.js` input-to-gesture layer, look-and-hold, Classic Simon, Simon Says
- `serve.py` local server + opens the browser
