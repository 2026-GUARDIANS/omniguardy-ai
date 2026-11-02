"""Frame-fed MediaPipe VIDEO-mode Silent Signal parity smoke test."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import cv2, mediapipe as mp
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/'hand_motion_desktop/src')]
from detector import _build_detector, MIN_FIST_FRAMES, REQUIRED_REPEATS, MAX_GAP_BETWEEN_REPS_FRAMES
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES, analyze_video_for_silent_signal_front
from gesture_rules import RepeatedFistToOpenTracker, frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front
VIDEO=ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4'; OUT=ROOT/'experiment_results/streaming_poc/phase3'
def run(target:int):
 cap=cv2.VideoCapture(str(VIDEO)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.; stride=max(1,round(fps/target)); d=_build_detector(); t=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES); index=hands=best=current=0; detected=None; times=[]; started=time.perf_counter()
 try:
  while True:
   ok,f=cap.read()
   if not ok:break
   if index%stride==0:
    s=time.perf_counter(); rgb=cv2.cvtColor(cv2.flip(f,1),cv2.COLOR_BGR2RGB); r=d.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(index*1000/fps)); times.append((time.perf_counter()-s)*1000); gs=[]
    for landmarks,handed in zip(r.hand_landmarks,r.handedness): gs.append(classify_gesture_front(landmarks,handed[0].category_name))
    hands+=int(bool(r.hand_landmarks)); g=frame_gesture_from_hands(gs); current=current+1 if g=='fist' else 0; best=max(best,current)
    if t.update(g) and detected is None: detected=index
   index+=1
 finally: cap.release();d.close()
 return {'targetFps':target,'sourceFps':fps,'processedFrames':len(times),'detected':detected is not None,'detectedFrame':detected,'handPresentSampleFrames':hands,'wallSeconds':round(time.perf_counter()-started,4),'meanMediaPipeMs':round(sum(times)/len(times),3),'rule':'unchanged frame-count thresholds: fist=5, transition=30 source-frame units'}
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True); baseline=analyze_video_for_silent_signal_front(VIDEO); result={'video':str(VIDEO),'baselineMp4':{'detected':baseline.detected,'detectedFrame':baseline.detected_frame},'online':[run(x) for x in (30,15,10)],'note':'15/10 retain original frame index timestamps but intentionally do not time-correct gesture thresholds.'};(OUT/'175204_silent_signal_online.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
