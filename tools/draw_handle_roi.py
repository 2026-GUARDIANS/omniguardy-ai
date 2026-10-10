"""Interactively select the normalized AUDIO door-handle ROI from one video frame."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.input))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("Cannot read requested frame")
    selected = cv2.selectROI("Draw handle-only ROI, then ENTER", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, width, height = selected
    if width <= 0 or height <= 0:
        raise SystemExit("No ROI selected")
    image_height, image_width = image.shape[:2]
    x0, y0 = x / image_width, y / image_height
    x1, y1 = (x + width) / image_width, (y + height) / image_height
    print(f"VISION_HANDLE_ROI={x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}")
    print("ROI must contain only the door handle; exclude doorbell/intercom areas.")


if __name__ == "__main__":
    main()
