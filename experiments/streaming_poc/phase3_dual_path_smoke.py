"""Phase 3 smoke: one timed source splits to MP4 writer and JPEG tracker.

No server/API is started.  This uses a recorded video as the single camera
source and writes only beneath experiment_results/streaming_poc/phase3.
"""
from __future__ import annotations
import json, queue, threading, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import cv2, pandas as pd, torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]; VIDEO=ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4'; OUT=ROOT/'experiment_results/streaming_poc/phase3'; MODEL=ROOT/'tracking/yolov8n.pt'
import sys
for p in (ROOT, ROOT/'tracking'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from tracking.keypad_rules_front import detect_front_rear_approach

SENTINEL=object(); JPEG_Q=85; ANALYSIS_SIZE=(960,540); MAX_YOLO_QUEUE=250

def worker_yolo(q:queue.Queue, state:dict[str,Any], source_size:tuple[int,int]) -> None:
    model=YOLO(str(MODEL)); device=0 if torch.cuda.is_available() else 'cpu'
    while True:
        item=q.get()
        if item is SENTINEL: q.task_done(); break
        started=time.perf_counter(); img=cv2.imdecode(item['jpeg'],cv2.IMREAD_COLOR)
        result=model.track(img,tracker='botsort.yaml',persist=True,save=False,conf=.3,classes=[0],stream=False,device=device,verbose=False,imgsz=640)[0]; ended=time.perf_counter()
        h,w=img.shape[:2]; sx,sy=source_size[0]/w,source_size[1]/h
        records=[]
        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None: continue
                x1,y1,x2,y2=box.xyxy[0].cpu().numpy(); records.append({'frame':item['frame'],'track_id':int(box.id[0]),'x1':float(x1*sx),'y1':float(y1*sy),'x2':float(x2*sx),'y2':float(y2*sy),'confidence':float(box.conf[0])})
        state['records'].extend(records); state['timings'].append({'frame':item['frame'],'enqueuedAt':item['enqueuedAt'],'startedAt':started,'endedAt':ended,'queueWaitMs':round((started-item['enqueuedAt'])*1000,3),'trackingMs':round((ended-started)*1000,3)}); q.task_done()
    for tr in list(getattr(getattr(model,'predictor',None),'trackers',[]) or []): tr.reset()
    state['finishedAt']=time.perf_counter()

def worker_writer(q:queue.Queue,state:dict[str,Any],fps:float,size:tuple[int,int],path:Path)->None:
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    while True:
        item=q.get()
        if item is SENTINEL: q.task_done(); break
        writer.write(item); state['frames']+=1; q.task_done()
    writer.release(); state['finishedAt']=time.perf_counter(); state['fileBytes']=path.stat().st_size if path.exists() else 0

def run(target_fps:int)->dict[str,Any]:
    OUT.mkdir(parents=True,exist_ok=True); cap=cv2.VideoCapture(str(VIDEO)); source_fps=cap.get(cv2.CAP_PROP_FPS) or 30.; size=(int(cap.get(3)),int(cap.get(4))); yq:queue.Queue=queue.Queue(); wq:queue.Queue=queue.Queue(); ys={'records':[],'timings':[],'finishedAt':None}; ws={'frames':0,'finishedAt':None,'fileBytes':0}; mp4=OUT/f'175204_recorded_30fps_analysis_{target_fps}fps.mp4'
    yt=threading.Thread(target=worker_yolo,args=(yq,ys,size),daemon=True); wt=threading.Thread(target=worker_writer,args=(wq,ws,source_fps,size,mp4),daemon=True); yt.start(); wt.start(); started=time.perf_counter(); index=produced=0; qlens=[]; aborted=False
    stride=max(1,round(source_fps/target_fps))
    while True:
        ok,frame=cap.read()
        if not ok: break
        scheduled=started+index/source_fps; delay=scheduled-time.perf_counter()
        if delay>0: time.sleep(delay)
        # Original 30-FPS path is always handed to recorder independently.
        wq.put(frame.copy())
        if index%stride==0:
            if yq.qsize()>=MAX_YOLO_QUEUE: aborted=True; break
            small=cv2.resize(frame,ANALYSIS_SIZE,interpolation=cv2.INTER_AREA); enc_started=time.perf_counter(); good,jpeg=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,JPEG_Q]); encoded=time.perf_counter()
            if not good: raise RuntimeError('jpeg encode failed')
            yq.put({'frame':index,'capturedAt':index/source_fps,'enqueuedAt':encoded,'jpeg':jpeg,'encodeMs':round((encoded-enc_started)*1000,3)}); produced+=1; qlens.append(yq.qsize())
        index+=1
    cap.release(); capture_done=time.perf_counter(); y_remaining=yq.qsize(); yq.put(SENTINEL); wq.put(SENTINEL); yt.join(); wt.join()
    track=pd.DataFrame(ys['records']); rear=detect_front_rear_approach(track,source_fps) if not track.empty else None
    result={'case':'175204','source':{'fps':source_fps,'resolution':size,'frames':index},'analysis':{'jpegResolution':ANALYSIS_SIZE,'jpegQuality':JPEG_Q,'targetFps':target_fps,'producedFrames':produced,'queueMax':max(qlens,default=0),'queueMean':round(sum(qlens)/len(qlens),3) if qlens else 0,'remainingAtCaptureEnd':y_remaining,'safetyAborted':aborted},'timing':{'captureSeconds':round(capture_done-started,4),'visionDoneSeconds':round(ys['finishedAt']-started,4),'visionPostCaptureSeconds':round(ys['finishedAt']-capture_done,4),'mp4DoneSeconds':round(ws['finishedAt']-started,4),'mp4PostCaptureSeconds':round(ws['finishedAt']-capture_done,4)},'mp4Recording':{'path':str(mp4),'framesWritten':ws['frames'],'fileBytes':ws['fileBytes']},'tracking':{'records':len(ys['records']),'trackedPersonCount':int(track.track_id.nunique()) if not track.empty else 0,'rearApproach':rear},'workerTiming':{'meanTrackingMs':round(sum(v['trackingMs'] for v in ys['timings'])/len(ys['timings']),3),'maxQueueWaitMs':max((v['queueWaitMs'] for v in ys['timings']),default=0),'firstTrackingStartOffsetMs':round((ys['timings'][0]['startedAt']-started)*1000,3) if ys['timings'] else None}}
    (OUT/f'175204_dual_{target_fps}fps.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/f'175204_dual_{target_fps}fps_worker_timing.json').write_text(json.dumps(ys['timings'],ensure_ascii=False,indent=2),encoding='utf-8'); return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fps',type=int,choices=[10,15,30],default=10);a=p.parse_args();print(json.dumps(run(a.fps),ensure_ascii=False))
