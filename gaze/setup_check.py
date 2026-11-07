"""Run once WITH internet before going to the ANT station:  python -m gaze.setup_check [camera index]

1. checks Python and package versions
2. downloads the MediaPipe face model into gaze/models (needed offline later)
3. opens the webcam, reports the resolution and frame rate it actually delivers,
   and checks that a face is found
"""
import sys
import time
import urllib.request

from .tracker import MODEL_PATH, MODEL_URL, Camera, FaceLandmarks, model_present


def ok(msg): print(f"  [ok]   {msg}")
def warn(msg): print(f"  [warn] {msg}")
def bad(msg): print(f"  [FAIL] {msg}")


def main():
    print("Gaze setup check")
    ok(f"Python {sys.version.split()[0]}")
    try:
        import importlib.metadata as md
        for pkg in ("mediapipe", "opencv-python", "numpy", "pyautogui"):
            ok(f"{pkg} {md.version(pkg)}")
    except Exception as e:
        bad(f"missing package ({e}). Run: .venv\\Scripts\\python -m pip install -r requirements-gaze.txt")
        return
    if not model_present():
        print("  downloading face model ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(MODEL_PATH)
        except Exception as e:
            bad(f"face model download failed ({e}). Connect to the internet and run this again.")
            return
    ok(f"face model at {MODEL_PATH}")

    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        cam = Camera(index)
    except RuntimeError as e:
        bad(str(e)); return
    faces = FaceLandmarks(video=True)
    found = n = 0
    t0 = time.time()
    while time.time() - t0 < 5:
        frame = cam.read()
        if frame is None:
            continue
        n += 1
        found += faces.find(frame) is not None
    elapsed = time.time() - t0
    cam.close(); faces.close()
    if n == 0:
        bad("camera returned no frames (is another app using it?)"); return
    fps = n / elapsed
    msg = f"camera {index}: {cam.size[0]}x{cam.size[1]}, about {fps:.0f} frames per second processed"
    (ok if fps >= 20 else warn)(msg + ("" if fps >= 20 else " - low; close other apps, add light (dim light lowers webcam fps)"))
    rate = found / n
    (ok if rate > 0.8 else bad)(f"face found in {rate:.0%} of frames" +
                                ("" if rate > 0.8 else " - face the camera, add front light"))
    print("Done. Start the cursor with:  .venv\\Scripts\\python -m gaze.app")


if __name__ == "__main__":
    main()
