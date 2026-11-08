"""The photosensitive-epilepsy warning. It is shown before SSVEP calibration (inside the calibration program itself, so it
cannot be skipped) and every time SSVEP is switched on in the controller."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

TITLE = "Photosensitive epilepsy warning"

BODY = (
    "SSVEP shows four squares that FLASH rapidly (8 to 15 times per second) in the corners of your screen.\n\n"
    "Flashing lights can trigger seizures in people with photosensitive epilepsy, including people who have never "
    "had a seizure before. Do NOT use SSVEP if you or anyone in your family has epilepsy or a history of seizures, "
    "or if you are sensitive to flashing or flickering lights.\n\n"
    "Stop immediately and look away if you feel dizzy, nauseous, get a headache, notice eye strain, a changed "
    "perception, or any unusual symptom. Take a break at least every 10 minutes and use SSVEP in a well-lit room.\n\n"
    "How to stop at any time:\n"
    "   • During calibration: press Esc.\n"
    "   • In the controller: click the SSVEP button again, or press Ctrl+Alt+S from any window.\n"
)


def confirm(parent, purpose: str) -> bool:
    """Modal warning. purpose: 'calibration' or 'enable'. True only if the user explicitly accepts."""
    what = "start SSVEP calibration" if purpose == "calibration" else "turn on SSVEP"
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(TITLE)
    box.setText(f"<b>Read this before you {what}.</b>")
    box.setInformativeText(BODY)
    go = box.addButton("I have read this and I am not at risk. Continue", QMessageBox.AcceptRole)
    cancel = box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    box.exec()
    return box.clickedButton() is go
