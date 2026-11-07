// Gesture -> browser input bindings.
// The EEG app turns each detected gesture into the OS input listed here (pyautogui), and the game
// only listens for that input. Head turns / gaze move the OS cursor (gaze app); winks are unlinked.
window.BINDINGS = {
  // Left button held at least this long = "clench hold". The controller only presses the button down
  // after its own hold_s (0.5 s), and sends a short clench as an instant click, so this can be short.
  clenchHoldMs: 300,
  eyebrowKeys: ["Space"],     // KeyboardEvent.code values sent for an eyebrow raise

  gestures: {
    eyebrow_raise: { label: "Eyebrow raise", input: "Space",           cue: "RAISE YOUR EYEBROWS" },
    jaw_clench:    { label: "Jaw clench",    input: "Left click",      cue: "CLENCH YOUR JAW" },
    clench_hold:   { label: "Clench hold",   input: "Hold left click", cue: "CLENCH AND HOLD" },
    double_blink:  { label: "Double blink",  input: "Right click",     cue: "DOUBLE BLINK" },
    head_left:     { label: "Head left",     input: "Cursor left",     cue: "TURN YOUR HEAD LEFT" },
    head_right:    { label: "Head right",    input: "Cursor right",    cue: "TURN YOUR HEAD RIGHT" },
    head_up:       { label: "Head up",       input: "Cursor up",       cue: "TILT YOUR HEAD UP" },
    head_down:     { label: "Head down",     input: "Cursor down",     cue: "TILT YOUR HEAD DOWN" },
  },

  // Keyboard stand-ins for testing without the cap (not used by the EEG app).
  debugKeys: {
    Digit1: "pad0", Digit2: "pad1", Digit3: "pad2", Digit4: "pad3",
    KeyJ: "jaw_clench", KeyH: "clench_hold", KeyB: "double_blink",
    ArrowLeft: "head_left", ArrowRight: "head_right", ArrowUp: "head_up", ArrowDown: "head_down",
  },
};
