"""Frame-state AUDIO parity smoke: original rules, no production edits."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import cv2, pandas as pd, torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'final_demo/src'))
from exhibition_audio_service import (ANALYSIS_SECONDS,EDGE_DENSITY_MAX,LAPLACIAN_VARIANCE_MAX,OCCLUSION_MIN_SUSTAINED_SECONDS,detect_camera_occlusion,detect_person_dwell)
OUT=ROOT/'experiment_results/streaming_poc/phase4';MODEL=ROOT/'tracking/yolov8n.pt'; FULL=ROOT/'final_demo/results_front_rules/runs/full64_20261020/videos'
CASES={'occlusion':'01_camera_occlusion/20261019_173225.mp4','dwell':'02_knock/exit_reentry/20261019_173843.mp4'}
def full(relative):
 for p in FULL.glob('*.json'):
  j=json.loads(p.read_text(encoding='utf-8'))
  if j.get('relative_path')==relative:return pd.DataFrame(j['full'].get('tracking',[]))
 raise FileNotFoundError(relative)
def online_occlusion(path,fps,small=None):
 cap=cv2.VideoCapture(str(path)); flags=[]; limit=max(1,round(ANALYSIS_SECONDS*fps)); i=0
 while i<limit:
  ok,f=cap.read()
  if not ok:break
  if small:f=cv2.resize(f,small,interpolation=cv2.INTER_AREA)
  g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); edge=float(cv2.countNonZero(cv2.Canny(g,100,200)))/g.size; lap=float(cv2.Laplacian(g,cv2.CV_64F).var());flags.append(edge<EDGE_DENSITY_MAX and lap<LAPLACIAN_VARIANCE_MAX);i+=1
 cap.release();required=max(1,round(OCCLUSION_MIN_SUSTAINED_SECONDS*fps));events=[];start=None
 for n,v in enumerate(flags+[False]):
  if v and start is None:start=n
  elif not v and start is not None:
   if n-start>=required:events.append({'eventType':'CAMERA_OCCLUSION_SUSPECTED','detectedFrame':start+required-1,'details':{'startSeconds':round(start/fps,3),'endSeconds':round((n-1)/fps,3),'durationSeconds':round((n-start)/fps,3)}})
   start=None
 return events
def online_track(path,fps):
 cap=cv2.VideoCapture(str(path));m=YOLO(str(MODEL));dev=0 if torch.cuda.is_available() else 'cpu';rows=[];i=0;limit=max(1,round(ANALYSIS_SECONDS*fps))
 while i<limit:
  ok,f=cap.read()
  if not ok:break
  r=m.track(f,tracker='botsort.yaml',persist=True,save=False,conf=.3,classes=[0],stream=False,device=dev,verbose=False,imgsz=640)[0]
  if r.boxes is not None:
   for b in r.boxes:
    if b.id is None:continue
    x1,y1,x2,y2=b.xyxy[0].cpu().numpy();rows.append({'frame':i,'track_id':int(b.id[0]),'x1':float(x1),'y1':float(y1),'x2':float(x2),'y2':float(y2),'confidence':float(b.conf[0])})
  i+=1
 cap.release();return pd.DataFrame(rows)
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);oc=ROOT/'final_demo/videos'/CASES['occlusion'];dw=ROOT/'final_demo/videos'/CASES['dwell'];fps=cv2.VideoCapture(str(oc)).get(cv2.CAP_PROP_FPS) or 30
 base_oc,_=detect_camera_occlusion(oc,fps);on_oc=online_occlusion(oc,fps);on_oc_960=online_occlusion(oc,fps,(960,540));dfps=cv2.VideoCapture(str(dw)).get(cv2.CAP_PROP_FPS) or 30;base_dw=detect_person_dwell(full(CASES['dwell']),dfps);t=time.perf_counter();online=online_track(dw,dfps);on_dw=detect_person_dwell(online,dfps)
 result={'occlusion':{'baseline':base_oc,'onlineOriginal':on_oc,'online960x540':on_oc_960},'dwell':{'baselineStoredFull64':base_dw,'online30fps':on_dw,'onlineTrackedPersonCount':int(online.track_id.nunique()) if not online.empty else 0,'onlineTrackingSeconds':round(time.perf_counter()-t,3)},'note':'No FPS sampling/time correction tested; online original applies exact 30-FPS frame thresholds.'};(OUT/'audio_parity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
