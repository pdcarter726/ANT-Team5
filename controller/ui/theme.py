"""Neutral light UI with NC State red as the accent. Darker red only for paused and errors."""
from pathlib import Path

PAPER = "#F5F6F8"        # window background
PAPER_DEEP = "#FFFFFF"   # panels, inputs, tiles
GRID = "#DDE1E6"         # borders, plot grid
INK = "#1F2328"
TRACE = "#44546A"        # EEG traces, meter fill below threshold
ACCENT = "#CC0000"       # NC State Wolfpack Red: selection, focus, gesture markers
ACCENT_SOFT = "#FBEAEA"  # hover / selected backgrounds
RED = "#990000"          # Reynolds Red: paused and errors
MUTED = "#656D76"
GOOD = "#1A7F37"

DISPLAY = "Segoe UI"
BODY = "Segoe UI"
_IC = (Path(__file__).parent / "icons").as_posix()

QSS = f"""
* {{ font-family: "{BODY}"; font-size: 10pt; color: {INK}; }}
QMainWindow, QWidget#root {{ background: {PAPER}; }}
QLabel#title {{ font-size: 16pt; font-weight: 600; }}
QWidget#statusCard {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; border-radius: 6px; }}
QWidget#statusCard[listening="true"] {{ background: transparent; border-color: transparent; }}
QWidget#statusCard[listening="true"] QLabel {{ color: {MUTED}; }}
QLabel#statusKey {{ font-size: 7pt; font-weight: 600; color: {MUTED}; }}
QLabel#statusVal {{ font-size: 8.5pt; font-weight: 600; }}
QLabel#section {{ font-size: 10.5pt; font-weight: 600; color: {MUTED}; }}
QLabel#lastAction {{ font-size: 13pt; font-weight: 600; }}
QLabel#bigSub {{ font-size: 10.5pt; color: {MUTED}; }}
QPushButton {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ background: #F0F2F5; border-color: #C3C9D1; }}
QPushButton:pressed {{ background: #E4E8EE; }}
QPushButton:focus {{ outline: none; border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border-color: {ACCENT}; font-weight: 600; padding: 9px 22px; }}
QPushButton#primary:hover {{ background: #B30000; border-color: #B30000; }}
QLabel#sessionTitle {{ font-size: 20pt; font-weight: 600; }}
QLabel#sessionState {{ font-size: 12pt; font-weight: 600; }}
QFrame#simPanel {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; border-radius: 8px; }}
QFrame#simPanel QPushButton#test:pressed {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#pause {{ font-weight: 600; padding: 8px 22px; }}
QPushButton#pause[paused="true"] {{ background: {RED}; color: white; border-color: {RED}; }}
QPushButton#eye {{ font-weight: 600; padding: 8px 18px; }}
QPushButton#eye[on="true"] {{ background: {INK}; color: white; border-color: {INK}; }}
QPushButton#icon, QPushButton#eye {{ padding: 0 14px 0 10px; min-height: 34px; max-height: 34px; font-weight: 600; }}
QFrame#sourceSwitch {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; border-radius: 5px; }}
QPushButton#sourceOption {{ background: transparent; border: none; border-radius: 3px; padding: 4px 9px; font-size: 8.5pt; font-weight: 600; color: {MUTED}; }}
QPushButton#sourceOption:hover {{ background: #F0F2F5; color: {INK}; }}
QPushButton#sourceOption:checked {{ background: #E6E8EC; color: {INK}; }}
QPushButton#test {{ text-align: left; padding: 8px 12px; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {GRID}; }}
QTabBar::tab {{ background: transparent; padding: 8px 16px; color: {MUTED}; border: none; }}
QTabBar::tab:selected {{ color: {INK}; font-weight: 600; border-bottom: 2px solid {ACCENT}; }}
QListWidget {{ background: transparent; border: none; }}
QListWidget::item {{ padding: 3px 0; border-bottom: 1px solid {GRID}; }}
QTableWidget {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; border-radius: 6px; outline: 0; }}
QTableWidget::item {{ border-bottom: 1px solid #EEF0F3; padding: 0 10px; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: {PAPER_DEEP}; border: none; border-bottom: 1px solid {GRID}; padding: 8px 10px;
    font-weight: 600; color: {MUTED}; }}
QTableWidget QComboBox {{
    border: 1px solid transparent;
    background: transparent;
    border-radius: 4px;
    margin: 0 8px 0 0;
    padding: 3px 26px 3px 8px;
}}
QTableWidget QComboBox:hover {{ border-color: {GRID}; }}
QTableWidget QComboBox:on {{ border-color: {ACCENT}; background: {PAPER_DEEP}; }}
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; border-radius: 6px; padding: 4px 8px; }}
QComboBox:hover, QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: #C3C9D1; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:on {{ border-color: {ACCENT}; }}
QComboBox {{ padding-right: 28px; min-height: 22px; }}
QComboBox::drop-down {{ border: none; width: 26px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({_IC}/down.svg); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; padding: 4px; outline: 0;
    selection-background-color: {ACCENT_SOFT}; selection-color: {INK}; }}
QComboBox QAbstractItemView::item {{ min-height: 26px; padding: 0 8px; border-radius: 4px; }}
QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected:hover {{ background: #F5C9C9; color: {INK}; }}
QAbstractSpinBox {{ padding: 4px 22px 4px 8px; min-height: 22px; min-width: 70px; max-width: 120px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ border: none; width: 20px; background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-arrow {{ image: url({_IC}/up.svg); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url({_IC}/down.svg); width: 10px; height: 10px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #B8BFC8; border-radius: 4px; background: {PAPER_DEEP}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({_IC}/check.svg); }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #C9CED6; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #A9B0BA; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; }}
QPlainTextEdit {{ background: {PAPER_DEEP}; border: 1px solid {GRID}; border-radius: 6px; font-family: "Consolas"; font-size: 9pt; color: {MUTED}; }}
QToolTip {{ background: {INK}; color: white; border: none; padding: 4px; }}
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
