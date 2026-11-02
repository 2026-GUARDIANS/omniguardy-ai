"""Single-JPEG-source, concurrent YOLO + MediaPipe KEYPAD session PoC."""
from __future__ import annotations
import json,queue,threading,time,sys
from pathlib import Path
import cv2,mediapipe as mp,pandas as pd,torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'hand_motion_desktop/src')]
from detector import _build_detector,MIN_FIST_FRAMES,REQUIRED_REPEATS,MAX_GAP_BETWEEN_REPS_FRAMES
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES
from gesture_rules import RepeatedFistToOpenTracker,frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front
from tracking.keypad_rules_front import detect_front_rear_approach
VIDEO=ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4';OUT=ROOT/'experiment_results/streaming_poc/phase5';MODEL=ROOT/'tracking/yolov8n.pt';S=object();SIZE=(960,540);Q=85
def yolo_worker(q,state,source_size):
 m=YOLO(str(MODEL));dev=0 if torch.cuda.is_available() else 'cpu'
 while True:
  x=q.get()
  if x is S:q.task_done();break
  st=time.perf_counter();im=cv2.imdecode(x['jpeg'],cv2.IMREAD_COLOR);r=m.track(im,tracker='botsort.yaml',persist=True,save=False,conf=.3,classes=[0],stream=False,device=dev,verbose=False,imgsz=640)[0];en=time.perf_counter();sx,sy=source_size[0]/im.shape[1],source_size[1]/im.shape[0]
  if r.boxes is not None:
   for b in r.boxes:
    if b.id is None:continue
    a,b1,c,d=b.xyxy[0].cpu().numpy();state['rows'].append({'frame':x['frame'],'track_id':int(b.id[0]),'x1':float(a*sx),'y1':float(b1*sy),'x2':float(c*sx),'y2':float(d*sy),'confidence':float(b.conf[0])})
  state['times'].append((st-x['queued'],en-st));q.task_done()
 state['done']=time.perf_counter()
def hand_worker(q,state,fps):
 d=_build_detector();t=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES);best=cur=0
 while True:
  x=q.get()
  if x is S:q.task_done();break
  st=time.perf_counter();im=cv2.imdecode(x['jpeg'],cv2.IMREAD_COLOR);rgb=cv2.cvtColor(cv2.flip(im,1),cv2.COLOR_BGR2RGB);r=d.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(x['frame']*1000/fps));gs=[classify_gesture_front(l,h[0].category_name) for l,h in zip(r.hand_landmarks,r.handedness)];g=frame_gesture_from_hands(gs);cur=cur+1 if g=='fist' else 0;best=max(best,cur)
  if t.update(g) and state['frame'] is None:state['frame']=x['frame']
  state['hands']+=int(bool(r.hand_landmarks));state['times'].append((st-x['queued'],time.perf_counter()-st));q.task_done()
 d.close();state['best']=best;state['done']=time.perf_counter()
def main():
 OUT.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(VIDEO));fps=cap.get(cv2.CAP_PROP_FPS) or 30.;source=(int(cap.get(3)),int(cap.get(4)));yq,hq=queue.Queue(),queue.Queue();ys={'rows':[],'times':[],'done':None};hs={'frame':None,'hands':0,'times':[],'best':0,'done':None};yt=threading.Thread(target=yolo_worker,args=(yq,ys,source),daemon=True);ht=threading.Thread(target=hand_worker,args=(hq,hs,fps),daemon=True);yt.start();ht.start();start=time.perf_counter();index=0;ql=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  wait=start+index/fps-time.perf_counter()
  if wait>0:time.sleep(wait)
  sm=cv2.resize(f,SIZE,interpolation=cv2.INTER_AREA);good,j=cv2.imencode('.jpg',sm,[cv2.IMWRITE_JPEG_QUALITY,Q]);now=time.perf_counter()
  if not good:raise RuntimeError('jpeg')
  item={'frame':index,'jpeg':j,'queued':now};yq.put(item);hq.put(item);ql.append((yq.qsize(),hq.qsize()));index+=1
 cap.release();capture=time.perf_counter();yr,hr=yq.qsize(),hq.qsize();yq.put(S);hq.put(S);yt.join();ht.join();tracking=pd.DataFrame(ys['rows']);rear=detect_front_rear_approach(tracking,fps) if not tracking.empty else None;events=[]
 if hs['frame'] is not None:events.append({'eventType':'SILENT_SIGNAL','confidence':round(min(1,hs['best']/MIN_FIST_FRAMES),2),'detectedFrame':hs['frame'],'details':{'signalCode':'S1','signalName':'emergency_hand_gesture'}})
 if rear:events.append(rear)
 done=max(ys['done'],hs['done']);result={'status':'success','result':{'video':{'fileName':VIDEO.name,'fps':round(fps,3),'frameCount':index},'behavior':None,'visionEvents':events,'observations':{'trackedPersonCount':int(tracking.track_id.nunique()) if not tracking.empty else 0,'hasTracking':not tracking.empty,'hasPose':hs['hands']>0}},'sessionMetrics':{'sourceFrames':index,'jpegResolution':SIZE,'jpegQuality':Q,'captureSeconds':round(capture-start,4),'finalJsonSeconds':round(done-start,4),'postCaptureDrainSeconds':round(done-capture,4),'queueMaxYolo':max((a for a,b in ql),default=0),'queueMaxMediaPipe':max((b for a,b in ql),default=0),'remainingYoloAtCaptureEnd':yr,'remainingMediaPipeAtCaptureEnd':hr,'yoloMeanMs':round(sum(x[1] for x in ys['times'])/len(ys['times'])*1000,3),'mediaPipeMeanMs':round(sum(x[1] for x in hs['times'])/len(hs['times'])*1000,3),'yoloMaxWaitMs':round(max(x[0] for x in ys['times'])*1000,3),'mediaPipeMaxWaitMs':round(max(x[0] for x in hs['times'])*1000,3)}}
 (OUT/'keypad_integrated_175204.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
