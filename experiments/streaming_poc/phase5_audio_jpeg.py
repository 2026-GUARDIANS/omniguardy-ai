"""Original-resolution JPEG quality-85 AUDIO parity smoke."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import cv2,pandas as pd,torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'final_demo/src')]
from exhibition_audio_service import detect_camera_occlusion,detect_person_dwell,ANALYSIS_SECONDS,EDGE_DENSITY_MAX,LAPLACIAN_VARIANCE_MAX,OCCLUSION_MIN_SUSTAINED_SECONDS
OUT=ROOT/'experiment_results/streaming_poc/phase5';MODEL=ROOT/'tracking/yolov8n.pt';O=ROOT/'final_demo/videos/01_camera_occlusion/20261019_173225.mp4';D=ROOT/'final_demo/videos/02_knock/exit_reentry/20261019_173843.mp4'
def occ(path,fps):
 c=cv2.VideoCapture(str(path));f=[];i=0;limit=round(ANALYSIS_SECONDS*fps)
 while i<limit:
  ok,x=c.read()
  if not ok:break
  ok,j=cv2.imencode('.jpg',x,[cv2.IMWRITE_JPEG_QUALITY,85]);x=cv2.imdecode(j,cv2.IMREAD_COLOR);g=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY);f.append(float(cv2.countNonZero(cv2.Canny(g,100,200)))/g.size<EDGE_DENSITY_MAX and float(cv2.Laplacian(g,cv2.CV_64F).var())<LAPLACIAN_VARIANCE_MAX);i+=1
 c.release();req=round(OCCLUSION_MIN_SUSTAINED_SECONDS*fps);out=[];s=None
 for n,v in enumerate(f+[False]):
  if v and s is None:s=n
  elif not v and s is not None:
   if n-s>=req:out.append(s+req-1)
   s=None
 return out
def dwell(path,fps):
 c=cv2.VideoCapture(str(path));m=YOLO(str(MODEL));dev=0 if torch.cuda.is_available() else 'cpu';rows=[];i=0
 while i<round(ANALYSIS_SECONDS*fps):
  ok,x=c.read()
  if not ok:break
  ok,j=cv2.imencode('.jpg',x,[cv2.IMWRITE_JPEG_QUALITY,85]);x=cv2.imdecode(j,cv2.IMREAD_COLOR);r=m.track(x,tracker='botsort.yaml',persist=True,save=False,conf=.3,classes=[0],stream=False,device=dev,verbose=False,imgsz=640)[0]
  if r.boxes is not None:
   for b in r.boxes:
    if b.id is None:continue
    a,b1,c1,d1=b.xyxy[0].cpu().numpy();rows.append({'frame':i,'track_id':int(b.id[0]),'x1':float(a),'y1':float(b1),'x2':float(c1),'y2':float(d1)})
  i+=1
 c.release();df=pd.DataFrame(rows);return detect_person_dwell(df,fps),int(df.track_id.nunique()) if not df.empty else 0
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);fo=cv2.VideoCapture(str(O)).get(cv2.CAP_PROP_FPS) or 30;fd=cv2.VideoCapture(str(D)).get(cv2.CAP_PROP_FPS) or 30;t=time.perf_counter();base,_=detect_camera_occlusion(O,fo);out={'occlusionBaselineFrames':[x['detectedFrame'] for x in base],'occlusionJpegFrames':occ(O,fo),'dwellJpeg':dwell(D,fd),'wallSeconds':round(time.perf_counter()-t,3),'note':'1920x1080 JPEG encode/decode quality 85; no network'};(OUT/'audio_jpeg_parity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
