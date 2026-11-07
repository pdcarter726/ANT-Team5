import pytest
import yaml

from input_layer import ActionExecutor, CursorController, VirtualBackend, load_mapping

GESTURES = {"left_wink", "right_wink", "double_blink", "eyebrow_raise", "jaw_clench",
            "head_left", "head_right", "head_up", "head_down"}


def test_default_mapping_validates_and_overrides():
    m = load_mapping(known_gestures=GESTURES)
    assert m.event_for("left_wink")[1]["action"] == "click"
    assert m.overrides()["jaw_clench"]["hold"] >= 3                 # pause gesture needs a held clench


def test_mapping_errors_are_reported_with_names(tmp_path):
    bad = tmp_path / "m.yaml"
    bad.write_text(yaml.safe_dump({"gestures": {"left_wink": "nope", "wiggle": None},
                                   "events": {"x": {"action": "explode"},
                                              "c": {"action": "cursor", "axis": "z", "dir": 1}}}))
    with pytest.raises(ValueError) as e:
        load_mapping(bad, known_gestures={"left_wink"})
    msg = str(e.value)
    assert "no event 'nope'" in msg and "explode" in msg and "axis" in msg and "wiggle" in msg


def test_executor_translates_actions():
    b = VirtualBackend()
    ex = ActionExecutor(b)
    ex.do({"action": "click", "button": "left", "count": 2})
    ex.do({"action": "key", "key": "tab"})
    ex.do({"action": "hotkey", "keys": ["alt", "tab"]})
    ex.do({"action": "scroll", "dy": -3})
    ex.do({"action": "pause_toggle"})                               # handled by the engine, no OS call
    assert b.actions() == [("click", "left", 2), ("key", "tab"), ("hotkey", ("alt", "tab")), ("scroll", -3)]


def run_cursor(cc, est, p_head, seconds, t0=0.0, dt=0.01):
    """update() every 0.125 s like the scorer, step() every 10 ms like the cursor thread."""
    t, next_upd = t0, t0
    for _ in range(int(round(seconds / dt))):
        if t >= next_upd - 1e-9:
            cc.update(est[0], est[1], p_head, t)
            next_upd += 0.125
        cc.step(t, dt)
        t += dt
    return t


def cfg_():
    return dict(max_speed_px_s=500, dead_zone=0.3, smoothing=0.0, accel_power=1.0, head_gate=0.5,
                velocity_tau_s=0.05, freeze_ms=300)


def test_dead_zone_and_gate_stop_movement():
    b = VirtualBackend()
    cc = CursorController(b, cfg_())
    run_cursor(cc, (0.25, -0.2), 0.9, 2.0)                          # inside the dead zone
    assert abs(b.x) < 1 and abs(b.y) < 1
    run_cursor(cc, (1.0, 1.0), 0.2, 2.0, t0=2.0)                    # strong estimate but not a head posture
    assert abs(b.x) < 1 and abs(b.y) < 1


def test_direction_and_speed_scale():
    b = VirtualBackend()
    cc = CursorController(b, cfg_())
    run_cursor(cc, (1.0, -1.0), 0.9, 2.0)
    assert b.x > 600 and b.y < -600                                 # ~500 px/s for ~1.9 s: x right, y up
    b2 = VirtualBackend()
    c2 = CursorController(b2, cfg_())
    run_cursor(c2, (0.65, 0.0), 0.9, 2.0)                           # halfway between dead zone and full
    assert 0.3 * b.x < b2.x < 0.7 * b.x


def test_click_freeze_stops_cursor_for_300ms_then_resumes():
    b = VirtualBackend()
    cc = CursorController(b, cfg_())
    t = run_cursor(cc, (1.0, 0.0), 0.9, 1.0)
    cc.freeze(t)
    x0 = b.x
    t = run_cursor(cc, (1.0, 0.0), 0.9, 0.28, t0=t)
    assert b.x == x0                                                # nothing moved during the freeze
    run_cursor(cc, (1.0, 0.0), 0.9, 1.0, t0=t)
    assert b.x > x0 + 100


def test_disabled_cursor_does_not_move():
    b = VirtualBackend()
    cc = CursorController(b, cfg_())
    cc.enabled = False
    run_cursor(cc, (1.0, 1.0), 0.9, 1.0)
    assert b.log == []
