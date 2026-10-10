"""Frozen temporal-candidate inference shared by FastAPI and field testing.

The feature builders deliberately come from ``evaluation.train_lstm_v2`` so
serving cannot silently diverge from the training representation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
# The experiment module predates package imports and imports its sibling helper
# by filename. Keep it importable from FastAPI without copying feature code.
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
from evaluation.train_lstm_v2 import (  # noqa: E402
    SEQ_COLUMNS, SUMMARY_COLUMNS, sequence, summary_pose, summary_tracking,
)
CANDIDATES = ROOT / "results" / "evaluation" / "lstm_v2_img" / "final_candidates"
RESPONSE_CLASSES = ["N1", "A17", "A18_20", "A19", "A21"]
SLOTS, STEP, PADDING = 30, 10, 0.12


class TemporalInference:
    def __init__(self, candidate: str) -> None:
        if candidate not in {"temporal_trial6", "temporal_trial6_scaleaug", "temporal_img"}:
            raise ValueError(f"Unknown temporal candidate: {candidate}")
        self.candidate = candidate
        self.folder = CANDIDATES / ({"temporal_trial6": "fusion_trial6", "temporal_trial6_scaleaug": "fusion_trial6_scaleaug", "temporal_img": "fusion_img"}[candidate])
        self.manifest = json.loads((self.folder / "serving_manifest.json").read_text(encoding="utf-8"))
        self.classes = list(self.manifest["classes"])
        self.session = self._onnx_session(self.folder / "model.onnx")
        self.image = candidate == "temporal_img"
        artifacts = joblib.load(self.folder / ("preprocessors.joblib" if self.image else "scalers.joblib"))
        self.sequence_scaler = artifacts["sequence_scaler"]
        self.summary_scaler = artifacts["summary_scaler"]
        if self.image:
            self.image_scaler, self.pca = artifacts["image_scaler"], artifacts["pca_64"]
            self._load_embedder()

    @staticmethod
    def _onnx_session(path: Path):
        import onnxruntime as ort
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def _load_embedder(self) -> None:
        # Explicit local weights prevents an unexpected download in production.
        weight_path = ROOT / self.manifest["image_embedding"]["weights_path"]
        if not weight_path.is_file():
            raise FileNotFoundError(f"ResNet18 weights are missing: {weight_path}")
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        model.fc = torch.nn.Identity()
        self.embedder, self.image_transform = model.eval(), weights.transforms(antialias=True)

    @staticmethod
    def make_temporal_inputs(tracking: pd.DataFrame, pose: pd.DataFrame, frame_count: int, start_frame: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the exact train_lstm_v2 30x12 sequence, mask, and 40 summary."""
        end = min(max(frame_count - 1, start_frame), start_frame + SLOTS * STEP - 1)
        tracking_window = tracking[(tracking.frame >= start_frame) & (tracking.frame <= end)] if "frame" in tracking else tracking
        pose_window = pose[(pose.frame >= start_frame) & (pose.frame <= end)] if "frame" in pose else pose
        x, mask = sequence(pose_window, tracking_window, start_frame, end)
        values = {**summary_tracking(tracking_window), **summary_pose(pose_window)}
        summary = np.asarray([[values.get(column, 0.0) for column in SUMMARY_COLUMNS]], dtype=np.float32)
        return x[None], mask[None], summary

    @staticmethod
    def _largest_boxes(tracking: pd.DataFrame, start_frame: int = 0) -> dict[int, pd.Series]:
        required = {"frame", "x1", "y1", "x2", "y2"}
        if not required.issubset(tracking.columns):
            return {}
        wanted = set(range(start_frame, start_frame + SLOTS * STEP, STEP))
        out: dict[int, pd.Series] = {}
        for frame, group in tracking[tracking.frame.isin(wanted)].groupby("frame"):
            area = (group.x2 - group.x1) * (group.y2 - group.y1)
            out[int(frame)] = group.loc[area.idxmax()]
        return out

    def _image_embeddings(self, video_path: Path, tracking: pd.DataFrame, start_frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Read the video once and collect crops for the fixed slot frames."""
        chosen = self._largest_boxes(tracking, start_frame)
        raw = np.zeros((SLOTS, 512), np.float32)
        present = np.zeros((SLOTS, 1), np.float32)
        if not chosen:
            return raw, present
        wanted, crops, slots = set(chosen), [], []
        capture = cv2.VideoCapture(str(video_path))
        frame = -1
        try:
            while wanted:
                ok, image = capture.read()
                if not ok:
                    break
                frame += 1
                if frame not in wanted:
                    continue
                row = chosen[frame]
                height, width = image.shape[:2]
                bw, bh = row.x2 - row.x1, row.y2 - row.y1
                x1, y1 = max(0, int(row.x1 - bw * PADDING)), max(0, int(row.y1 - bh * PADDING))
                x2, y2 = min(width, int(row.x2 + bw * PADDING)), min(height, int(row.y2 + bh * PADDING))
                crop = image[y1:y2, x1:x2]
                wanted.remove(frame)
                if crop.size:
                    crops.append(torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
                    slots.append((frame - start_frame) // STEP)
        finally:
            capture.release()
        if crops:
            batch = torch.stack([self.image_transform(crop) for crop in crops])
            with torch.no_grad():
                vectors = self.embedder(batch).float().numpy()
            raw[slots] = vectors
            present[slots, 0] = 1.0
        return raw, present

    def predict(self, video_path: Path, tracking: pd.DataFrame, pose: pd.DataFrame, frame_count: int) -> tuple[dict[str, float], dict[str, float]]:
        started = time.perf_counter()
        sequence_raw, pose_mask, summary_raw = self.make_temporal_inputs(tracking, pose, frame_count)
        sequence_scaled = self.sequence_scaler.transform(sequence_raw.reshape(-1, len(SEQ_COLUMNS))).reshape(sequence_raw.shape).astype(np.float32)
        summary_scaled = self.summary_scaler.transform(summary_raw).astype(np.float32)
        timings = {"input": time.perf_counter() - started}
        if self.image:
            started = time.perf_counter()
            image_raw, image_present = self._image_embeddings(video_path, tracking)
            image_pca = self.pca.transform(self.image_scaler.transform(image_raw)).astype(np.float32)
            image_pca *= image_present
            model_sequence = np.concatenate([sequence_scaled[0], image_pca, image_present], axis=1)[None].astype(np.float32)
            model_mask = image_present[None].astype(np.float32)
            timings["image_embedding"] = time.perf_counter() - started
        else:
            model_sequence, model_mask = sequence_scaled, pose_mask.astype(np.float32)
        started = time.perf_counter()
        # Frozen candidate ONNX files were exported with the 8-row parity
        # batch. Repeat a single request to that fixed batch and retain row 0.
        # This preserves the supplied artifact without re-exporting it.
        batch = int(self.session.get_inputs()[0].shape[0])
        feed = {
            "sequence": np.repeat(model_sequence, batch, axis=0),
            "mask": np.repeat(model_mask, batch, axis=0),
            "summary": np.repeat(summary_scaled, batch, axis=0),
        }
        logits = self.session.run(["logits"], feed)[0][0]
        probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
        timings["model"] = time.perf_counter() - started
        mapped = {label: float(probabilities[self.classes.index(label)]) for label in RESPONSE_CLASSES}
        return mapped, timings
