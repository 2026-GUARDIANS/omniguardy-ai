"""Local-only Silent Signal EARLY_EVENT callback proof of concept.

No existing Vision API is imported or changed.  A localhost-only mock receiver,
MediaPipe frame worker, optional YOLO backlog worker, and callback dispatcher
run in separate threads.  The event state machine uses the existing default
frame-level gesture rule unchanged.
"""
from __future__ import annotations
import json, os, queue, sys, threading, time, urllib.request, urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2, mediapipe as mp, torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[3]; sys.path[:0]=[str(ROOT),str(ROOT/'hand_motion_desktop/src')]
from detector import _build_detector,MIN_FIST_FRAMES,REQUIRED_REPEATS,MAX_GAP_BETWEEN_REPS_FRAMES
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES
from gesture_rules import RepeatedFistToOpenTracker,frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front
OUT=ROOT/'experiment_results/streaming_poc/early_event';POS=ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4';NEG_DIR=ROOT/'final_demo/videos/03_keypad/static_two_people';MODEL=ROOT/'tracking/yolov8n.pt';S=object();SIZE=(960,540);QUALITY=85

class Receiver(BaseHTTPRequestHandler):
    received=[]; fail_first=True
    def do_POST(self):
        data=self.rfile.read(int(self.headers.get('Content-Length','0'))); payload=json.loads(data); now=time.perf_counter(); Receiver.received.append({'receivedAtPerf':now,'payload':payload})
        status=500 if Receiver.fail_first and len(Receiver.received)==1 else 204
        self.send_response(status);self.end_headers()
    def log_message(self,*args): pass

def iso_now(): return datetime.now().astimezone().isoformat(timespec='milliseconds')
def callback_worker(q,endpoint,records):
    while True:
        job=q.get()
        if job is S:q.task_done();break
        for attempt in (1,2):
            start=time.perf_counter(); code=None; error=None
            try:
                req=urllib.request.Request(endpoint,data=json.dumps(job['payload']).encode(),headers={'Content-Type':'application/json'},method='POST')
                with urllib.request.urlopen(req,timeout=2) as response: code=response.status
            except urllib.error.HTTPError as exc: code=exc.code; error=repr(exc)
            except Exception as exc: error=repr(exc)
            end=time.perf_counter();records.append({'eventInstanceId':job['payload']['eventInstanceId'],'sequence':job['payload']['sequence'],'attempt':attempt,'startedAtPerf':start,'endedAtPerf':end,'status':code,'error':error})
            if code and 200<=code<300:break
            time.sleep(.05)
        q.task_done()
def media_worker(q,callback_q,state,fps,capture_start):
    detector=_build_detector();tracker=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES);sequence=0
    while True:
        item=q.get()
        if item is S:q.task_done();break
        started=time.perf_counter();im=cv2.imdecode(item['jpeg'],cv2.IMREAD_COLOR);rgb=cv2.cvtColor(cv2.flip(im,1),cv2.COLOR_BGR2RGB);result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(item['frame']*1000/fps));gestures=[classify_gesture_front(l,h[0].category_name) for l,h in zip(result.hand_landmarks,result.handedness)];triggered=tracker.update(frame_gesture_from_hands(gestures));done=time.perf_counter()
        state['frames'].append({'frame':item['frame'],'capturedAtVideoSeconds':item['frame']/fps,'inputAtPerf':item['inputAt'],'startedAtPerf':started,'doneAtPerf':done,'queueAgeMs':round((started-item['inputAt'])*1000,3),'mediaPipeMs':round((done-started)*1000,3),'gestures':gestures,'triggered':triggered})
        if triggered:
            sequence+=1; emitted=time.perf_counter(); instance=f"{state['sessionId']}-signal-{sequence:03d}"; payload={'messageType':'EARLY_EVENT','sessionId':state['sessionId'],'eventInstanceId':instance,'sequence':sequence,'provisional':True,'event':{'eventType':'SILENT_SIGNAL','detectedFrame':item['frame'],'confirmedFrame':item['frame']},'emittedAt':iso_now()};state['events'].append({'payload':payload,'confirmedFrame':item['frame'],'confirmedInputAtPerf':item['inputAt'],'mediaPipeDoneAtPerf':done,'emittedAtPerf':emitted});callback_q.put({'payload':payload})
        q.task_done()
    detector.close();state['doneAtPerf']=time.perf_counter()
def yolo_worker(q,state):
    model=YOLO(str(MODEL));dev=0 if torch.cuda.is_available() else 'cpu'
    while True:
        item=q.get()
        if item is S:q.task_done();break
        im=cv2.imdecode(item['jpeg'],cv2.IMREAD_COLOR);model.track(im,tracker='botsort.yaml',persist=True,save=False,conf=.3,classes=[0],stream=False,device=dev,verbose=False,imgsz=640);state['processed']+=1;q.task_done()
    state['doneAtPerf']=time.perf_counter()
def replay_positive():
    Receiver.received=[];Receiver.fail_first=True;server=ThreadingHTTPServer(('127.0.0.1',0),Receiver);endpoint=os.getenv('VISION_EARLY_CALLBACK_URL',f'http://127.0.0.1:{server.server_port}/callback')
    if not endpoint.startswith('http://127.0.0.1:'):raise ValueError('PoC permits localhost callback only')
    threading.Thread(target=server.serve_forever,daemon=True).start();cap=cv2.VideoCapture(str(POS));fps=cap.get(cv2.CAP_PROP_FPS) or 30.;mq,yq,cq=queue.Queue(),queue.Queue(),queue.Queue();state={'sessionId':'early-poc-175204','frames':[],'events':[]};ys={'processed':0};callbacks=[];mt=threading.Thread(target=media_worker,args=(mq,cq,state,fps,time.perf_counter()),daemon=True);yt=threading.Thread(target=yolo_worker,args=(yq,ys),daemon=True);ct=threading.Thread(target=callback_worker,args=(cq,endpoint,callbacks),daemon=True);mt.start();yt.start();ct.start();start=time.perf_counter();index=0
    while True:
        ok,f=cap.read()
        if not ok:break
        delay=start+index/fps-time.perf_counter()
        if delay>0:time.sleep(delay)
        sm=cv2.resize(f,SIZE,interpolation=cv2.INTER_AREA);ok,j=cv2.imencode('.jpg',sm,[cv2.IMWRITE_JPEG_QUALITY,QUALITY]);now=time.perf_counter();item={'frame':index,'jpeg':j,'inputAt':now};mq.put(item);yq.put(item);index+=1
    cap.release();capture_end=time.perf_counter();mq.put(S);yq.put(S);mt.join();yt.join();cq.put(S);ct.join();server.shutdown();server.server_close()
    final_at=time.perf_counter();event=state['events'][0] if state['events'] else None; first_receive=Receiver.received[0] if Receiver.received else None; success_receive=Receiver.received[-1] if Receiver.received else None
    ages=[x['queueAgeMs'] for x in state['frames']]; media=[x['mediaPipeMs'] for x in state['frames']]
    final={'messageType':'FINAL_RESULT','sessionId':state['sessionId'],'relatedEarlyEventInstanceIds':[x['payload']['eventInstanceId'] for x in state['events']],'status':'success','result':{'visionEvents':[{'eventType':x['payload']['event']['eventType'],'detectedFrame':x['payload']['event']['detectedFrame']} for x in state['events']]},'note':'PoC envelope only; it is not a replacement for the complete legacy Vision result assembly.'}
    return {'sessionId':state['sessionId'],'endpoint':endpoint,'sourceFrames':index,'captureSeconds':round(capture_end-start,4),'yoloProcessedFrames':ys['processed'],'yoloDoneSeconds':round(ys['doneAtPerf']-start,4),'mediaDoneSeconds':round(state['doneAtPerf']-start,4),'finalResultGeneratedSeconds':round(final_at-start,4),'finalResultPrototype':final,'mediaPipeQueue':{'processedFrames':len(state['frames']),'maxFrameAgeMs':round(max(ages),3),'meanFrameAgeMs':round(sum(ages)/len(ages),3),'meanMediaPipeMs':round(sum(media)/len(media),3)},'earlyEvents':state['events'],'callbackAttempts':callbacks,'receiverRequests':Receiver.received,'latency':None if not event else {'captureStartToConfirmedInputMs':round((event['confirmedInputAtPerf']-start)*1000,3),'confirmedInputToMediaDoneMs':round((event['mediaPipeDoneAtPerf']-event['confirmedInputAtPerf'])*1000,3),'mediaDoneToEmitMs':round((event['emittedAtPerf']-event['mediaPipeDoneAtPerf'])*1000,3),'captureStartToFirstReceiverMs':round((first_receive['receivedAtPerf']-start)*1000,3),'captureStartToSuccessfulReceiverMs':round((success_receive['receivedAtPerf']-start)*1000,3),'confirmedInputToSuccessfulReceiverMs':round((success_receive['receivedAtPerf']-event['confirmedInputAtPerf'])*1000,3)}}
def negative_check(video,ground_truth):
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30.;d=_build_detector();t=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES);i=0;hits=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        rgb=cv2.cvtColor(cv2.flip(cv2.resize(f,SIZE),1),cv2.COLOR_BGR2RGB);r=d.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(i*1000/fps));g=frame_gesture_from_hands([classify_gesture_front(l,h[0].category_name) for l,h in zip(r.hand_landmarks,r.handedness)]);
        if t.update(g):hits.append(i)
        i+=1
    cap.release();d.close();return {'video':str(video),'frames':i,'triggeredFrames':hits,'groundTruthStatus':ground_truth}
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);result={'positive':replay_positive(),'humanReviewedNegatives':[negative_check(NEG_DIR/'20261019_174439.mp4','Human-reviewed Silent Signal negative (prior validation)'),negative_check(NEG_DIR/'20261019_174454.mp4','Human-reviewed Silent Signal negative (prior validation)')],'notes':['detectedFrame and confirmedFrame are identical because the legacy state machine returns true at condition satisfaction','callback first attempt deliberately returns local HTTP 500; retry preserves eventInstanceId and sequence','no Spring Boot/network latency measured']};(OUT/'early_callback_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
