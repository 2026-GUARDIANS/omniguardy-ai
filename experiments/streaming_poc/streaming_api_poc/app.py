"""Isolated Streaming API PoC: online Silent Signal EARLY_EVENT only.

Run on a port other than 8000.  This module never imports or alters
api.vision_api.  It is deliberately not a replacement for the complete MP4
Vision API: AUDIO, Rear Approach, and complete legacy final-event assembly are
not included yet.
"""
from __future__ import annotations

import json, os, queue, sys, threading, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "hand_motion_desktop/src")]
sys.path.insert(0, str(ROOT / "final_demo/src"))
from detector import _build_detector, MIN_FIST_FRAMES, REQUIRED_REPEATS, MAX_GAP_BETWEEN_REPS_FRAMES
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES
from gesture_rules import RepeatedFistToOpenTracker, frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front
from ultralytics import YOLO
import torch
from experiments.streaming_poc.streaming_api_poc.adapters import (
    OnlineOcclusionAdapter, final_audio_events, final_keypad_events, source_is_contiguous,
)

app = FastAPI(title="OmniGuardy Streaming Silent Signal PoC", version="0.1")
END = object()
ALLOWED_CALLBACK_HOSTS = {x.strip() for x in os.getenv("STREAMING_POC_CALLBACK_HOSTS", "127.0.0.1,localhost").split(",") if x.strip()}
CALLBACK_BEARER_TOKEN = os.getenv("STREAMING_POC_CALLBACK_BEARER_TOKEN")
CALLBACK_KEY = os.getenv("STREAMING_POC_CALLBACK_KEY")
QUEUE_CAPACITY = max(1, int(os.getenv("STREAMING_POC_QUEUE_CAPACITY", "300")))
COMPLETED_SESSION_TTL_SECONDS = max(1.0, float(os.getenv("STREAMING_POC_COMPLETED_SESSION_TTL_SECONDS", "3600")))


class StartRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=128)
    triggerId: str | None = Field(default=None, max_length=128)
    # Legacy PoC alias only. New Pi/Spring Boot callers must use triggerId.
    eventId: str | None = Field(default=None, max_length=128)
    securityEventId: str | None = Field(default=None, max_length=128)
    triggerType: str = Field(default="KEYPAD", max_length=32)
    sourceFps: float = Field(gt=0, le=120)
    callbackUrl: str | None = None
    finalResultCallbackEnabled: bool = False


class CompleteRequest(BaseModel):
    sessionId: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def validate_callback(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in ALLOWED_CALLBACK_HOSTS:
        raise HTTPException(400, "callbackUrl host is not in STREAMING_POC_CALLBACK_HOSTS allowlist")
    return url


class Session:
    def __init__(self, request: StartRequest):
        self.id, self.trigger_id, self.trigger_type, self.fps = request.sessionId, request.triggerId or request.eventId, request.triggerType, request.sourceFps
        self.security_event_id = request.securityEventId
        self.final_callback_enabled = request.finalResultCallbackEnabled
        self.callback_url = validate_callback(request.callbackUrl)
        self.q: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self.lock = threading.RLock(); self.closed = False; self.expected_frame = 0; self.events: list[dict[str, Any]] = []
        self.created_monotonic = time.monotonic(); self.completed_monotonic: float | None = None
        self.callback_records: list[dict[str, Any]] = []; self.worker_done = threading.Event(); self.final_dispatched = False; self.final_payload: dict[str, Any] | None = None; self.error: str | None = None; self.max_queue_depth = 0
        self.dispatcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"callback-{self.id}")
        self.worker = threading.Thread(target=self._run, daemon=True, name=f"mediapipe-{self.id}"); self.worker.start()

    def push(self, frame_index: int, source_frame_index: int, captured_at: str | None, jpeg: bytes) -> str:
        with self.lock:
            if self.closed: raise HTTPException(409, "session input is already complete")
            if frame_index < self.expected_frame: return "duplicate"
            if frame_index > self.expected_frame: raise HTTPException(409, f"out-of-order frame: expected {self.expected_frame}")
            try: self.q.put_nowait({"frame": frame_index, "sourceFrame": source_frame_index, "capturedAt": captured_at, "jpeg": jpeg, "receivedAt": time.perf_counter()})
            except queue.Full: raise HTTPException(429, "session queue is full; retry this frame")
            self.max_queue_depth = max(self.max_queue_depth, self.q.qsize()); self.expected_frame += 1
            return "accepted"

    def complete(self) -> None:
        with self.lock:
            if not self.closed:
                self.closed = True
                threading.Thread(target=self._enqueue_end, args=(self.q,), daemon=True, name=f"complete-{self.id}").start()

    @staticmethod
    def _enqueue_end(target: queue.Queue[dict[str, Any] | object]) -> None:
        # Blocking here preserves every accepted frame, but it is deliberately
        # outside the HTTP handler so /vision/complete can return 202 promptly.
        target.put(END)

    def release(self) -> None:
        """Release callback resources only after a terminal session TTL expires."""
        self.dispatcher.shutdown(wait=False, cancel_futures=False)

    def _callback(self, payload: dict[str, Any]) -> None:
        if not self.callback_url: return
        # Spring Boot requires the callback key. Keep localhost/legacy Bearer
        # callbacks usable, but do not send an unauthenticated request to the
        # production Spring Boot host when its required key is absent.
        callback_host = urllib.parse.urlparse(self.callback_url).hostname
        if callback_host == "api.omniguardy.cloud" and not CALLBACK_KEY:
            with self.lock:
                self.callback_records.append({"messageType": payload["messageType"], "eventInstanceId": payload.get("eventInstanceId"), "attempt": 0, "status": None, "error": "callback authentication is not configured: STREAMING_POC_CALLBACK_KEY is required for api.omniguardy.cloud", "at": now_iso(), "elapsedMs": 0.0})
            return
        for attempt in (1, 2):
            started = time.perf_counter(); status = None; error = None
            try:
                headers = {"Content-Type": "application/json"}
                if CALLBACK_BEARER_TOKEN: headers["Authorization"] = f"Bearer {CALLBACK_BEARER_TOKEN}"
                if CALLBACK_KEY: headers["X-Vision-Callback-Key"] = CALLBACK_KEY
                req = urllib.request.Request(self.callback_url, data=json.dumps(payload).encode(), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=2) as response: status = response.status
            except urllib.error.HTTPError as exc:
                status = exc.code
                error = (f"callback authentication rejected (HTTP {status}; credentials redacted)"
                         if status in {401, 403} else f"callback returned HTTP {status}")
            except Exception as exc:
                # Exception text can include request headers on malformed
                # credentials, so never retain it in a status/log record.
                error = f"callback transport error ({type(exc).__name__}; credentials redacted)"
            with self.lock: self.callback_records.append({"messageType": payload["messageType"], "eventInstanceId": payload.get("eventInstanceId"), "attempt": attempt, "status": status, "error": error, "at": now_iso(), "elapsedMs": round((time.perf_counter()-started)*1000,3)})
            if status and 200 <= status < 300: return
            time.sleep(.05)

    def _emit_early(self, frame: int, captured_at: str | None) -> None:
        sequence = len(self.events) + 1; iid = f"{self.id}-signal-{sequence:03d}"
        event = {"eventType": "SILENT_SIGNAL", "detectedFrame": frame, "confirmedFrame": frame, "capturedAt": captured_at, "eventInstanceId": iid}
        self.events.append(event)
        payload = {"messageType": "EARLY_EVENT", "sessionId": self.id, "triggerId": self.trigger_id, "securityEventId": self.security_event_id, "triggerType": self.trigger_type, "eventInstanceId": iid, "sequence": sequence, "provisional": True, "event": {k:event[k] for k in ("eventType","detectedFrame","confirmedFrame","capturedAt")}, "emittedAt": now_iso()}
        self.dispatcher.submit(self._callback, payload)

    def _run(self) -> None:
        detector = _build_detector(); tracker = RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS, min_fist_frames=MIN_FIST_FRAMES, max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES, max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES)
        try:
            while True:
                item = self.q.get()
                if item is END: self.q.task_done(); break
                frame = int(item["sourceFrame"]); image = cv2.imdecode(np.frombuffer(item["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None: self.q.task_done(); continue
                rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(frame * 1000 / self.fps))
                gestures = [classify_gesture_front(lm, hand[0].category_name) for lm, hand in zip(result.hand_landmarks, result.handedness)]
                if tracker.update(frame_gesture_from_hands(gestures)):
                    with self.lock: self._emit_early(frame, item["capturedAt"])
                self.q.task_done()
        except Exception as exc:
            with self.lock: self.error = repr(exc)
        finally:
            detector.close()
            with self.lock: self.completed_monotonic = time.monotonic()
            self.worker_done.set(); self._dispatch_final()

    def _dispatch_final(self) -> None:
        with self.lock:
            if self.final_dispatched: return
            self.final_dispatched = True
            if not self.final_callback_enabled: return
            final = self.result_payload()
        self.dispatcher.submit(self._callback, final)

    def result_payload(self) -> dict[str, Any]:
        # Compatibility shape is retained where possible, but this deliberately
        # contains only the online Silent Signal PoC event family.
        if self.final_payload is not None: return self.final_payload
        status = "failed" if self.error else "success"
        payload={"messageType": "FINAL_RESULT", "sessionId": self.id, "trigger": {"triggerId": self.trigger_id, "triggerType": self.trigger_type, "securityEventId": self.security_event_id}, "relatedEarlyEventInstanceIds": [x["eventInstanceId"] for x in self.events], "status": status, "result": {"visionEvents": [{k:v for k,v in e.items() if k != "eventInstanceId"} for e in self.events], "trackedPersonCount": None, "streamingPocScope": "SILENT_SIGNAL only"}, "error": self.error, "emittedAt": now_iso()}
        if self.worker_done.is_set(): self.final_payload=payload
        return payload

    def status(self) -> dict[str, Any]:
        with self.lock:
            analysis_state = "FAILED" if self.error else ("COMPLETED" if self.worker_done.is_set() else ("PROCESSING" if self.closed else "RECEIVING"))
            ttl_remaining = None if self.completed_monotonic is None else max(0.0, COMPLETED_SESSION_TTL_SECONDS - (time.monotonic() - self.completed_monotonic))
            return {"sessionId": self.id, "triggerId": self.trigger_id, "securityEventId": self.security_event_id, "triggerType": self.trigger_type, "analysisState": analysis_state, "inputComplete": self.closed, "nextExpectedFrameIndex": self.expected_frame, "queueDepth": self.q.qsize(), "mediaPipeQueueMaxDepth": self.max_queue_depth, "yoloQueueMaxDepth": None, "finalResultAvailable": self.worker_done.is_set(), "completedSessionTtlSecondsRemaining": None if ttl_remaining is None else round(ttl_remaining, 3), "earlyEventCount": len(self.events), "callbackAttempts": list(self.callback_records), "error": self.error}


class PhaseBSession(Session):
    """Session-local MediaPipe, YOLO/BoT-SORT, and optional AUDIO workers."""
    def __init__(self, request: StartRequest):
        self.yolo_q: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self.audio_q: queue.Queue[dict[str, Any] | object] | None = queue.Queue(maxsize=QUEUE_CAPACITY) if request.triggerType == "AUDIO" else None
        self.done_workers: set[str] = set(); self.expected_workers = {"media", "yolo"} | ({"audio"} if self.audio_q else set())
        self.tracking_rows: list[dict[str, Any]] = []; self.source_indices: list[int] = []; self.frame_width: int | None = None; self.frame_height: int | None = None
        self.occlusion = OnlineOcclusionAdapter() if self.audio_q else None; self.final_events: list[dict[str, Any]] = []; self.tracked_person_count: int | None = None; self.analysis_completeness = "PROCESSING"
        self.yolo_max_queue = 0; self.audio_max_queue = 0
        super().__init__(request)
        self.yolo_worker = threading.Thread(target=self._run_yolo, daemon=True, name=f"yolo-{self.id}"); self.yolo_worker.start()
        self.audio_worker = None
        if self.audio_q:
            self.audio_worker = threading.Thread(target=self._run_audio, daemon=True, name=f"audio-{self.id}"); self.audio_worker.start()

    def push(self, frame_index: int, source_frame_index: int, captured_at: str | None, jpeg: bytes) -> str:
        with self.lock:
            if self.closed: raise HTTPException(409, "session input is already complete")
            if frame_index < self.expected_frame: return "duplicate"
            if frame_index > self.expected_frame: raise HTTPException(409, f"out-of-order frame: expected {self.expected_frame}")
            queues=[self.q,self.yolo_q]+([self.audio_q] if self.audio_q else [])
            if any(q.qsize() >= q.maxsize for q in queues): raise HTTPException(429, "session queue is full; retry this frame")
            item={"frame":frame_index,"sourceFrame":source_frame_index,"capturedAt":captured_at,"jpeg":jpeg,"receivedAt":time.perf_counter()}
            for q in queues:q.put_nowait(item)
            self.max_queue_depth=max(self.max_queue_depth,self.q.qsize()); self.yolo_max_queue=max(self.yolo_max_queue,self.yolo_q.qsize()); self.audio_max_queue=max(self.audio_max_queue,self.audio_q.qsize() if self.audio_q else 0)
            self.source_indices.append(source_frame_index); self.expected_frame+=1
            return "accepted"

    def complete(self) -> None:
        with self.lock:
            if self.closed:return
            self.closed=True
            targets=[self.q,self.yolo_q]+([self.audio_q] if self.audio_q else [])
            for target in targets:
                threading.Thread(target=self._enqueue_end,args=(target,),daemon=True,name=f"complete-{self.id}").start()

    def _run(self) -> None:
        detector=_build_detector(); tracker=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES)
        try:
            while True:
                item=self.q.get()
                if item is END:self.q.task_done();break
                frame=int(item["sourceFrame"]); image=cv2.imdecode(np.frombuffer(item["jpeg"],dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is not None:
                    rgb=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB); result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(frame*1000/self.fps)); gestures=[classify_gesture_front(lm,h[0].category_name) for lm,h in zip(result.hand_landmarks,result.handedness)]
                    if tracker.update(frame_gesture_from_hands(gestures)):
                        with self.lock:self._emit_early(frame,item["capturedAt"])
                self.q.task_done()
        except Exception as exc:
            with self.lock:self.error=repr(exc)
        finally:
            detector.close(); self._mark_done("media")

    def _run_yolo(self) -> None:
        model=None
        try:
            model=YOLO(str(ROOT/"tracking/yolov8n.pt")); device=0 if torch.cuda.is_available() else "cpu"; imgsz=max(160,int(os.getenv("VISION_YOLO_IMGSZ","640")))
            while True:
                item=self.yolo_q.get()
                if item is END:self.yolo_q.task_done();break
                image=cv2.imdecode(np.frombuffer(item["jpeg"],dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is not None:
                    if self.frame_width is None:self.frame_height,self.frame_width=image.shape[:2]
                    result=model.track(image,tracker="botsort.yaml",persist=True,save=False,conf=.3,classes=[0],stream=False,device=device,verbose=False,imgsz=imgsz)[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            if box.id is None:continue
                            x1,y1,x2,y2=box.xyxy[0].cpu().numpy(); self.tracking_rows.append({"frame":int(item["sourceFrame"]),"track_id":int(box.id[0]),"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),"confidence":float(box.conf[0])})
                self.yolo_q.task_done()
        except Exception as exc:
            with self.lock:self.error=repr(exc)
        finally:self._mark_done("yolo")

    def _run_audio(self) -> None:
        try:
            assert self.audio_q is not None and self.occlusion is not None
            while True:
                item=self.audio_q.get()
                if item is END:self.audio_q.task_done();break
                image=cv2.imdecode(np.frombuffer(item["jpeg"],dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is not None:self.occlusion.observe(int(item["sourceFrame"]),image)
                self.audio_q.task_done()
        except Exception as exc:
            with self.lock:self.error=repr(exc)
        finally:self._mark_done("audio")

    def _mark_done(self,name:str)->None:
        with self.lock:
            self.done_workers.add(name)
            if self.done_workers != self.expected_workers:return
            contiguous=source_is_contiguous(self.source_indices)
            if self.error:self.analysis_completeness="FAILED"
            elif not contiguous:self.analysis_completeness="PARTIAL_SAMPLED_TIME_AXIS"
            elif self.trigger_type=="KEYPAD":
                extras,count=final_keypad_events(self.tracking_rows,self.fps,self.frame_width,self.frame_height,True); self.final_events=[*self.events,*extras]; self.tracked_person_count=count; self.analysis_completeness="COMPLETE_KEYPAD"
            elif self.trigger_type=="AUDIO":
                assert self.occlusion is not None
                extras,count=final_audio_events(self.tracking_rows,self.occlusion,self.fps,self.frame_width,self.frame_height,True); self.final_events=[*extras]; self.tracked_person_count=count; self.analysis_completeness="COMPLETE_AUDIO"
            else:self.analysis_completeness="FAILED"; self.error=f"unsupported triggerType={self.trigger_type}"
            if not self.final_events and self.trigger_type=="KEYPAD":self.final_events=list(self.events)
            self.completed_monotonic=time.monotonic(); self.worker_done.set(); self._dispatch_final()

    def result_payload(self)->dict[str,Any]:
        if self.final_payload is not None:return self.final_payload
        status="failed" if self.error else ("partial" if self.analysis_completeness.startswith("PARTIAL") else "success")
        payload={"messageType":"FINAL_RESULT","sessionId":self.id,"trigger":{"triggerId":self.trigger_id,"triggerType":self.trigger_type,"securityEventId":self.security_event_id},"analysisState":"FAILED" if self.error else "COMPLETED","analysisCompleteness":self.analysis_completeness,"relatedEarlyEventInstanceIds":[x["eventInstanceId"] for x in self.events],"status":status,"result":{"visionEvents":[{k:v for k,v in e.items() if k!="eventInstanceId"} for e in self.final_events],"trackedPersonCount":self.tracked_person_count,"streamingPocScope":"KEYPAD Silent+Rear or AUDIO Occlusion+Dwell when COMPLETE"},"error":self.error,"emittedAt":now_iso()}
        if self.worker_done.is_set():self.final_payload=payload
        return payload

    def status(self)->dict[str,Any]:
        base=super().status(); base.update({"analysisState":"FAILED" if self.error else ("COMPLETED" if self.worker_done.is_set() else ("PROCESSING" if self.closed else "RECEIVING")),"analysisCompleteness":self.analysis_completeness,"mediaPipeQueueMaxDepth":self.max_queue_depth,"yoloQueueMaxDepth":self.yolo_max_queue,"audioQueueMaxDepth":self.audio_max_queue,"trackingRecordCount":len(self.tracking_rows)}); return base


sessions: dict[str, Session] = {}; sessions_lock = threading.RLock()

def cleanup_expired_sessions() -> int:
    """Remove only terminal sessions after their result-retention TTL.

    Active RECEIVING/PROCESSING sessions are never removed by this PoC helper;
    production needs a separately agreed input-idle timeout policy.
    """
    now=time.monotonic()
    with sessions_lock:
        expired=[sid for sid,session in sessions.items() if session.worker_done.is_set() and session.completed_monotonic is not None and now-session.completed_monotonic>=COMPLETED_SESSION_TTL_SECONDS]
        removed=[sessions.pop(sid) for sid in expired]
    for session in removed: session.release()
    return len(removed)

@app.post("/vision/start", status_code=201)
def start(request: StartRequest):
    cleanup_expired_sessions()
    if not (request.triggerId or request.eventId): raise HTTPException(422, "triggerId is required (eventId is accepted only as a legacy alias)")
    if request.triggerType == "AUDIO" and not request.securityEventId: raise HTTPException(422, "securityEventId is required for AUDIO")
    with sessions_lock:
        if request.sessionId in sessions: raise HTTPException(409, "sessionId already exists")
        sessions[request.sessionId] = PhaseBSession(request)
    return {"sessionId": request.sessionId, "triggerId": request.triggerId or request.eventId, "status": "started", "nextExpectedFrameIndex": 0, "scope": "online Silent Signal EARLY_EVENT PoC"}

@app.post("/vision/frame", status_code=202)
async def frame(sessionId: str = Form(...), frameIndex: int = Form(...), sourceFrameIndex: int | None = Form(None), capturedAt: str | None = Form(None), frame: UploadFile = File(...)):
    cleanup_expired_sessions()
    if frame.content_type not in {"image/jpeg", "image/jpg"}: raise HTTPException(415, "frame must be image/jpeg")
    jpeg = await frame.read()
    with sessions_lock: session = sessions.get(sessionId)
    if not session: raise HTTPException(404, "unknown sessionId")
    source_index = frameIndex if sourceFrameIndex is None else sourceFrameIndex
    state = session.push(frameIndex, source_index, capturedAt, jpeg)
    return {"sessionId": sessionId, "frameIndex": frameIndex, "sourceFrameIndex": source_index, "status": state}

@app.post("/vision/complete", status_code=202)
def complete(request: CompleteRequest):
    cleanup_expired_sessions()
    with sessions_lock: session = sessions.get(request.sessionId)
    if not session: raise HTTPException(404, "unknown sessionId")
    session.complete()
    return {"sessionId": request.sessionId, "status": "input_complete", "resultUrl": f"/vision/result/{request.sessionId}"}

@app.get("/vision/status/{session_id}")
def status(session_id: str):
    cleanup_expired_sessions()
    with sessions_lock: session = sessions.get(session_id)
    if not session: raise HTTPException(404, "unknown sessionId")
    return session.status()

@app.get("/vision/result/{session_id}")
def result(session_id: str):
    cleanup_expired_sessions()
    with sessions_lock: session = sessions.get(session_id)
    if not session: raise HTTPException(404, "unknown sessionId")
    if not session.worker_done.is_set(): raise HTTPException(202, "analysis is still draining")
    return session.result_payload()
