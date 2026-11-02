"""Phase 2: bounded FPS/resolution/JPEG/backlog benchmark for streaming PoC.

This is not a streaming API.  It reads a recorded MP4 as a virtual camera and
tests the Vision-side work only.  Every run creates a new YOLO predictor, so
BoT-SORT state cannot cross experiment sessions.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "final_demo" / "videos"
FULL64 = ROOT / "final_demo" / "results_front_rules" / "runs" / "full64_20261020" / "videos"
MODEL = ROOT / "tracking" / "yolov8n.pt"
OUT = ROOT / "experiment_results" / "streaming_poc" / "phase2"
CASES = {
    "175204": "03_keypad/approach_signal/20261019_175204.mp4",
    "174955": "03_keypad/approach_signal/20261019_174955.mp4",
}
JPEG_QUALITY = 85
MAX_QUEUE = 250


def percentile(values: list[float], point: float) -> float | None:
    return round(float(np.percentile(values, point)), 5) if values else None


def find_stored(relative_path: str) -> list[dict[str, Any]]:
    for path in FULL64.glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("relative_path") == relative_path:
            return item["full"].get("tracking", [])
    raise FileNotFoundError(relative_path)


def selected(index: int, source_fps: float, target_fps: int) -> bool:
    # Source videos are about 30 FPS. Integer sampling preserves the original
    # source frameIndex/capture timestamp exactly for 30/15/10 PoC conditions.
    stride = max(1, round(source_fps / target_fps))
    return index % stride == 0


def resize(frame: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) == resolution:
        return frame
    return cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)


def normalize(row: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    return {**row, "nx1": row["x1"] / width, "ny1": row["y1"] / height,
            "nx2": row["x2"] / width, "ny2": row["y2"] / height}


def normalized_stored(rows: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    return [normalize(row, width, height) for row in rows]


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1, y1 = max(a["nx1"], b["nx1"]), max(a["ny1"], b["ny1"])
    x2, y2 = min(a["nx2"], b["nx2"]), min(a["ny2"], b["ny2"])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    area_a = max(0., a["nx2"] - a["nx1"]) * max(0., a["ny2"] - a["ny1"])
    area_b = max(0., b["nx2"] - b["nx1"]) * max(0., b["ny2"] - b["ny1"])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.


def compare(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    by_ref: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in reference: by_ref[int(row["frame"])].append(row)
    for row in candidate: by_candidate[int(row["frame"])].append(row)
    hits = same_ids = 0; scores: list[float] = []; mapping: dict[tuple[int, int], int] = defaultdict(int)
    for frame, refs in by_ref.items():
        available = list(by_candidate.get(frame, []))
        for ref in refs:
            if not available: continue
            best = max(available, key=lambda value: iou(ref, value)); score = iou(ref, best)
            if score < .5: continue
            available.remove(best); hits += 1; scores.append(score)
            same_ids += int(int(ref["track_id"]) == int(best["track_id"]))
            mapping[(int(ref["track_id"]), int(best["track_id"]))] += 1
    return {"referenceBoxes": len(reference), "candidateBoxes": len(candidate), "iouMatchesAt0_5": hits,
            "meanIou": round(sum(scores) / len(scores), 5) if scores else None,
            "sameTrackIdRate": round(same_ids / hits, 5) if hits else None,
            "trackIdMappings": [{"mp4TrackId": a, "onlineTrackId": b, "frames": n}
                                for (a, b), n in sorted(mapping.items())]}


def continuity(rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    by_id: dict[int, list[int]] = defaultdict(list)
    for row in rows: by_id[int(row["track_id"])].append(int(row["frame"]))
    return [{"trackId": track_id, "startFrame": min(frames), "endFrame": max(frames),
             "observedFrames": len(frames), "maxInternalGapFrames": max((b-a-1 for a,b in zip(frames, frames[1:])), default=0)}
            for track_id, frames in sorted(by_id.items())]


class Tracker:
    def __init__(self) -> None:
        self.model = YOLO(str(MODEL)); self.device: int | str = 0 if torch.cuda.is_available() else "cpu"

    def track(self, frame: np.ndarray, source_index: int, source_width: int, source_height: int) -> list[dict[str, Any]]:
        result = self.model.track(frame, tracker="botsort.yaml", persist=True, save=False, conf=.3,
                                  classes=[0], stream=False, device=self.device, verbose=False, imgsz=640)[0]
        records: list[dict[str, Any]] = []
        if result.boxes is not None:
            height, width = frame.shape[:2]
            for box in result.boxes:
                if box.id is None: continue
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                # Normalize to the original source dimensions, allowing resized
                # candidate bboxes to be compared without pixel-coordinate bias.
                records.append({"frame": source_index, "track_id": int(box.id[0]),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "confidence": float(box.conf[0]), "nx1": float(x1)/width, "ny1": float(y1)/height,
                    "nx2": float(x2)/width, "ny2": float(y2)/height,
                    "sourceWidth": source_width, "sourceHeight": source_height})
        return records

    def reset(self) -> None:
        for tracker in list(getattr(getattr(self.model, "predictor", None), "trackers", []) or []): tracker.reset()


def offline_run(case: str, target_fps: int, resolution: tuple[int,int], jpeg: bool) -> dict[str, Any]:
    relative = CASES[case]; source = VIDEOS / relative; cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.; sw,sh = int(cap.get(3)),int(cap.get(4)); tracker = Tracker()
    records: list[dict[str,Any]]=[]; inference: list[float]=[]; enc: list[float]=[]; dec: list[float]=[]; sizes: list[int]=[]; index=0; processed=0
    started=time.perf_counter()
    try:
        while True:
            ok, frame=cap.read()
            if not ok: break
            if selected(index, fps, target_fps):
                payload=resize(frame,resolution)
                if jpeg:
                    t=time.perf_counter(); ok_jpeg, encoded=cv2.imencode('.jpg',payload,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY]); enc.append(time.perf_counter()-t)
                    if not ok_jpeg: raise RuntimeError('JPEG encode failed')
                    sizes.append(int(encoded.nbytes)); t=time.perf_counter(); payload=cv2.imdecode(encoded,cv2.IMREAD_COLOR); dec.append(time.perf_counter()-t)
                t=time.perf_counter(); records.extend(tracker.track(payload,index,sw,sh)); inference.append(time.perf_counter()-t); processed+=1
            index+=1
    finally:
        cap.release(); tracker.reset()
    elapsed=time.perf_counter()-started; reference=[row for row in normalized_stored(find_stored(relative),sw,sh) if selected(int(row['frame']),fps,target_fps)]
    return {"kind":"offline", "case":case,"input":{"sourceFps":fps,"sourceResolution":[sw,sh],"targetFps":target_fps,"inputResolution":list(resolution),"jpeg":jpeg,"jpegQuality":JPEG_QUALITY if jpeg else None},
            "frames":{"sourceRead":index,"processed":processed}, "timing":{"wallSeconds":round(elapsed,4),"trackingFps":round(processed/elapsed,4),"inferenceMeanMs":round(statistics.mean(inference)*1000,4),"inferenceMedianMs":round(statistics.median(inference)*1000,4),"inferenceP95Ms":percentile([x*1000 for x in inference],95),"jpegEncodeMeanMs":round(statistics.mean(enc)*1000,4) if enc else None,"jpegDecodeMeanMs":round(statistics.mean(dec)*1000,4) if dec else None},
            "jpeg":{"averageBytes":round(statistics.mean(sizes),2) if sizes else None,"p95Bytes":percentile(sizes,95),"estimatedBytesPerSecondAtTargetFps":round(statistics.mean(sizes)*target_fps,2) if sizes else None},
            "comparisonToFull64":compare(reference,records),"continuity":continuity(records),"records":records}


def realtime_run(case:str,target_fps:int,resolution:tuple[int,int]) -> dict[str,Any]:
    relative=CASES[case]; cap=cv2.VideoCapture(str(VIDEOS/relative)); source_fps=cap.get(cv2.CAP_PROP_FPS) or 30.; sw,sh=int(cap.get(3)),int(cap.get(4)); q:queue.Queue[Any]=queue.Queue(); lock=threading.Lock(); state={"records":[],"items":[],"queueLengths":[],"aborted":False,"workerDone":None}; sentinel=object(); tracker=Tracker()
    def worker() -> None:
        while True:
            item=q.get()
            if item is sentinel: q.task_done(); break
            started=time.perf_counter(); decoded=cv2.imdecode(item['encoded'],cv2.IMREAD_COLOR); decoded_at=time.perf_counter(); rows=tracker.track(decoded,item['frame'],sw,sh); finished=time.perf_counter()
            with lock:
                state['records'].extend(rows); state['items'].append({"frame":item['frame'],"capturedAt":item['capturedAt'],"enqueuedAt":item['enqueuedAt'],"processingStartedAt":started,"processingEndedAt":finished,"encodeSeconds":item['encodeSeconds'],"decodeSeconds":decoded_at-started})
            q.task_done()
        tracker.reset(); state['workerDone']=time.perf_counter()
    thread=threading.Thread(target=worker,name='poc-tracker',daemon=True); thread.start(); capture_start=time.perf_counter(); index=0; produced=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if selected(index,source_fps,target_fps):
            scheduled=capture_start+produced/target_fps; remaining=scheduled-time.perf_counter()
            if remaining>0: time.sleep(remaining)
            if q.qsize()>=MAX_QUEUE: state['aborted']=True; break
            payload=resize(frame,resolution); encoded_started=time.perf_counter(); ok_jpeg,encoded=cv2.imencode('.jpg',payload,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
            if not ok_jpeg: raise RuntimeError('JPEG encode failed')
            enqueued=time.perf_counter(); q.put({"frame":index,"capturedAt":index/source_fps,"enqueuedAt":enqueued,"encoded":encoded,"encodeSeconds":enqueued-encoded_started}); state['queueLengths'].append(q.qsize()); produced+=1
        index+=1
    cap.release(); capture_finished=time.perf_counter(); remaining_at_capture_end=q.qsize(); q.put(sentinel); thread.join(); finished=state['workerDone'] or time.perf_counter()
    waits=[row['processingStartedAt']-row['enqueuedAt'] for row in state['items']]; runs=[row['processingEndedAt']-row['processingStartedAt'] for row in state['items']]
    reference=[row for row in normalized_stored(find_stored(relative),sw,sh) if selected(int(row['frame']),source_fps,target_fps)]
    return {"kind":"realtime_no_drop","case":case,"input":{"sourceFps":source_fps,"targetFps":target_fps,"sourceResolution":[sw,sh],"inputResolution":list(resolution),"jpeg":True,"jpegQuality":JPEG_QUALITY},"frames":{"sourceReadUntilStop":index,"produced":produced,"processed":len(state['items']),"remainingAtCaptureEnd":remaining_at_capture_end,"queueSafetyLimit":MAX_QUEUE,"safetyAborted":state['aborted']},"timing":{"captureSeconds":round(capture_finished-capture_start,4),"captureToAllTrackingDoneSeconds":round(finished-capture_start,4),"postCaptureDrainSeconds":round(finished-capture_finished,4),"effectiveTrackingFps":round(len(state['items'])/(finished-capture_start),4),"meanQueueWaitMs":round(statistics.mean(waits)*1000,4) if waits else None,"maxQueueWaitMs":round(max(waits)*1000,4) if waits else None,"meanTrackingMs":round(statistics.mean(runs)*1000,4) if runs else None},"queue":{"maxLength":max(state['queueLengths'],default=0),"meanLength":round(statistics.mean(state['queueLengths']),3) if state['queueLengths'] else 0},"comparisonToFull64":compare(reference,state['records']),"continuity":continuity(state['records'])}


def save(result:dict[str,Any],name:str) -> None:
    OUT.mkdir(parents=True,exist_ok=True); (OUT/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=OUT/'summary.csv'; flat={"name":name,"kind":result['kind'],"case":result['case'],"targetFps":result['input']['targetFps'],"resolution":"x".join(map(str,result['input']['inputResolution'])),"jpeg":result['input']['jpeg'],"wallSeconds":result.get('timing',{}).get('wallSeconds',result.get('timing',{}).get('captureToAllTrackingDoneSeconds')),"trackingFps":result.get('timing',{}).get('trackingFps',result.get('timing',{}).get('effectiveTrackingFps')),"maxBacklog":result.get('queue',{}).get('maxLength'),"sameTrackIdRate":result['comparisonToFull64'].get('sameTrackIdRate'),"meanIou":result['comparisonToFull64'].get('meanIou'),"jpegAverageBytes":result.get('jpeg',{}).get('averageBytes')}
    exists=summary.exists()
    with summary.open('a',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat));
        if not exists: writer.writeheader()
        writer.writerow(flat)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['offline','realtime'],required=True); parser.add_argument('--case',choices=CASES,default='175204'); parser.add_argument('--fps',type=int,choices=[10,15,30],required=True); parser.add_argument('--resolution',choices=['1920x1080','960x540','640x360'],required=True); parser.add_argument('--jpeg',action='store_true')
    args=parser.parse_args(); resolution=tuple(map(int,args.resolution.split('x'))); result=realtime_run(args.case,args.fps,resolution) if args.mode=='realtime' else offline_run(args.case,args.fps,resolution,args.jpeg); uses_jpeg=args.jpeg or args.mode=='realtime'; name=f'{args.case}_{args.mode}_{args.fps}fps_{args.resolution}_{"jpeg" if uses_jpeg else "direct"}'; save(result,name); print(json.dumps({"name":name,"timing":result['timing']},ensure_ascii=False))

if __name__=='__main__': main()
