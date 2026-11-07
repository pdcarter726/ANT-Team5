"""EEG chart paper: pale gridded paper, blue ink traces, red hand annotations."""
PAPER = "#EDF0EA"
PAPER_DEEP = "#E3E8E0"
GRID = "#D2D9D0"
INK = "#1C2836"
TRACE = "#2C4E7A"
RED = "#C3313B"
MUTED = "#6A7684"
GOOD = "#2F7D5B"

DISPLAY = "Bahnschrift"          # ships with Windows 10/11; instrument-panel lettering
BODY = "Segoe UI"

QSS = f"""
* {{ font-family: "{BODY}"; font-size: 10.5pt; color: {INK}; }}
QMainWindow, QWidget#root {{ background: {PAPER}; }}
QLabel#title {{ font-family: "{DISPLAY}"; font-size: 17pt; font-weight: 600; }}
QLabel#statusKey {{ color: {MUTED}; }}
QLabel#statusVal {{ font-family: "{DISPLAY}"; font-size: 11pt; }}
QLabel#section {{ font-family: "{DISPLAY}"; font-size: 12pt; color: {MUTED}; }}
QLabel#bigAction {{ font-family: "{DISPLAY}"; font-size: 34pt; font-weight: 600; color: {INK}; }}
QLabel#bigSub {{ font-size: 11pt; color: {MUTED}; }}
QPushButton {{ background: {PAPER}; border: 1.5px solid {INK}; border-radius: 3px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {RED}; color: {RED}; }}
QPushButton:focus {{ outline: none; border-color: {TRACE}; }}
QPushButton#pause {{ font-family: "{DISPLAY}"; font-size: 12pt; padding: 8px 22px; }}
QPushButton#pause[paused="true"] {{ background: {RED}; color: {PAPER}; border-color: {RED}; }}
QPushButton#test {{ border: 1px solid {GRID}; background: {PAPER_DEEP}; text-align: left; padding: 8px 12px; }}
QPushButton#test:hover {{ border-color: {RED}; }}
QTabWidget::pane {{ border: none; border-top: 1.5px solid {INK}; }}
QTabBar::tab {{ background: transparent; padding: 8px 18px; font-family: "{DISPLAY}"; font-size: 11.5pt; color: {MUTED}; border: none; }}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 3px solid {RED}; }}
QListWidget {{ background: transparent; border: none; font-size: 10.5pt; }}
QListWidget::item {{ padding: 3px 0; border-bottom: 1px solid {GRID}; }}
QTableWidget {{ background: {PAPER}; border: none; gridline-color: {GRID}; }}
QHeaderView::section {{ background: transparent; border: none; border-bottom: 1.5px solid {INK}; padding: 6px; font-family: "{DISPLAY}"; color: {MUTED}; }}
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {{ background: {PAPER}; border: 1px solid {GRID}; border-radius: 3px; padding: 4px 8px; }}
QComboBox:focus, QLineEdit:focus {{ border-color: {TRACE}; }}
QPlainTextEdit {{ background: {PAPER_DEEP}; border: none; font-size: 9.5pt; color: {MUTED}; }}
QToolTip {{ background: {INK}; color: {PAPER}; border: none; padding: 4px; }}
"""


def nice(gesture):
    """jaw_clench -> Jaw clench"""
    return gesture.replace("_", " ").capitalize()


def describe(action):
    """Action string -> plain words for people."""
    kind, _, arg = (action or "none").partition(":")
    words = {
        "move": lambda a: f"Move cursor {a}",
        "mouse": lambda a: f"{a.capitalize() if a else 'Left'} click (hold to drag)",
        "key": lambda a: f"Press {a}",
        "hotkey": lambda a: f"Press {a.replace('+', ' + ')}",
        "click": lambda a: {"left": "Left click", "right": "Right click", "double": "Double click",
                            "middle": "Middle click"}.get(a, f"{a} click"),
        "scroll": lambda a: "Scroll up" if a.lstrip("+").isdigit() and int(a) > 0 else "Scroll down",
        "hold": lambda a: f"Hold {a}",
        "type": lambda a: f"Type “{a}”",
        "menu": lambda a: "Quick bar",
        "eye": lambda a: {"pause": "Pause eye cursor", "resume": "Resume eye cursor", "toggle": "Pause/resume eye cursor",
                          "recenter": "Recenter eye cursor", "check": "Eye accuracy check",
                          "fix": "Apply eye correction"}.get(a, f"Eye cursor {a}"),
        "controller": lambda a: "Pause/resume controller",
        "none": lambda a: "Nothing",
    }
    try:
        return words.get(kind.strip().lower(), lambda a: action)(arg.strip())
    except Exception:
        return action
