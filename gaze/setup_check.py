"""Run once WITH internet before going to the ANT station:  python -m gaze.setup_check

1. checks Python and package versions
2. downloads the MediaPipe face model into gaze/models (EyeGestures would otherwise
   try to download it on every new machine, which fails without internet)
3. opens the webcam and checks that a face is found
"""
import sys
import time
import urllib.request

from .tracker import MODEL_PATH, MODEL_URL, use_local_model


def ok(msg): print(f"  [ok]   {msg}")
def bad(msg): print(f"  [FAIL] {msg}")


def main():
    print("Gaze setup check")
    if sys.version_info < (3, 13):
        bad(f"Python {sys.version.split()[0]}: need 3.13+ (eyeGestures 4.x only ships for 3.13)")
        return
    ok(f"Python {sys.version.split()[0]}")
    try:
        import importlib.metadata as md
        v = md.version("eyeGestures")
        from eyeGestures.face import FaceFinder  # noqa: F401
        ok(f"eyeGestures {v}")
    except Exception as e:
        bad(f"eyeGestures 4.x not usable ({e}). Run: pip install -r requirements-gaze.txt")
        return
    if not use_local_model():
        print("  downloading face model ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    if not use_local_model():
        bad("face model download failed"); return
    ok(f"face model at {MODEL_PATH}")

    import cv2
    import numpy as np
    from .tracker import Camera
    from eyeGestures.face import FaceFinder
    try:
        cam = Camera(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    except RuntimeError as e:
        bad(str(e)); return
    finder = FaceFinder(str(MODEL_PATH))
    found, n, t0 = 0, 0, time.time()
    while time.time() - t0 < 4:
        frame = cam.read()
        if frame is None:
            continue
        n += 1
        det = finder.find(frame)
        found += bool(det and det.face_landmarks)
    cam.close()
    if n == 0:
        bad("camera returned no frames"); return
    ok(f"camera {n / 4:.0f} fps")
    rate = found / n
    (ok if rate > 0.8 else bad)(f"face found in {rate:.0%} of frames" +
                                ("" if rate > 0.8 else " - face the camera, add front light"))
    print("Done. Start the cursor with:  python -m gaze.app")


if __name__ == "__main__":
    main()
