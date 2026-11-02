"""7-second real-time HTTP replay for the standalone Silent Signal API PoC."""
import argparse, json, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2, requests

ROOT=Path(__file__).resolve().parents[3];DEFAULT_VIDEO=ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4'
class Receiver(BaseHTTPRequestHandler):
 received=[]
 def do_POST(self):
  Receiver.received.append({'receivedPerf':time.perf_counter(),'payload':json.loads(self.rfile.read(int(self.headers['Content-Length'])))});self.send_response(204);self.end_headers()
 def log_message(self,*a):pass
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fps',type=float,required=True);ap.add_argument('--base',default='http://127.0.0.1:8011');ap.add_argument('--video',type=Path,default=DEFAULT_VIDEO);ap.add_argument('--trigger-type',default='KEYPAD',choices=['KEYPAD','AUDIO']);ap.add_argument('--width',type=int,default=960);ap.add_argument('--height',type=int,default=540);ap.add_argument('--output-dir',type=Path,default=ROOT/'experiment_results/streaming_poc/streaming_api_poc/phase_c');a=ap.parse_args();Receiver.received=[]
 server=ThreadingHTTPServer(('127.0.0.1',0),Receiver);threading.Thread(target=server.serve_forever,daemon=True).start();sid=f'fps-{int(a.fps)}-{uuid.uuid4().hex[:8]}'
 cap=cv2.VideoCapture(str(a.video));source_fps=cap.get(cv2.CAP_PROP_FPS) or 30.;start=time.perf_counter();body={'sessionId':sid,'triggerId':f'fps-test-{a.fps}','triggerType':a.trigger_type,'sourceFps':source_fps,'callbackUrl':f'http://127.0.0.1:{server.server_port}/callback'}
 if a.trigger_type=='AUDIO':body['securityEventId']=f'local-audio-{sid}'
 requests.post(a.base+'/vision/start',json=body,timeout=5).raise_for_status()
 source_index=0;transmit=0;accepted=duplicate=conflict=full=0;next_source=0.;limit=int(7*source_fps)
 while source_index<limit:
  ok,frame=cap.read()
  if not ok:break
  # Round to the nearest source frame.  With 30.004 source FPS and a nominal
  # 30 FPS transmit request, ceil-like comparison would incorrectly skip 1,3…
  # and turn a full-cadence test into a sampled-time-axis test.
  if source_index>=round(next_source):
   target=start+transmit/a.fps;delay=target-time.perf_counter()
   if delay>0:time.sleep(delay)
   small=cv2.resize(frame,(a.width,a.height));_,jpg=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,85])
   r=requests.post(a.base+'/vision/frame',data={'sessionId':sid,'frameIndex':str(transmit),'sourceFrameIndex':str(source_index),'capturedAt':f'{source_index/source_fps:.6f}'},files={'frame':('frame.jpg',jpg.tobytes(),'image/jpeg')},timeout=5)
   if r.status_code==202:
    accepted+=1
    if r.json()['status']=='duplicate':duplicate+=1
   elif r.status_code==409:conflict+=1
   elif r.status_code==429:full+=1
   else:r.raise_for_status()
   transmit+=1;next_source+=source_fps/a.fps
  source_index+=1
 cap.release();capture_end=time.perf_counter();requests.post(a.base+'/vision/complete',json={'sessionId':sid},timeout=5).raise_for_status();status={}
 for _ in range(100):
  status=requests.get(a.base+f'/vision/status/{sid}',timeout=5).json()
  if status['finalResultAvailable']:break
  time.sleep(.1)
 result=requests.get(a.base+f'/vision/result/{sid}',timeout=5).json();done=time.perf_counter();server.shutdown();server.server_close()
 early=next((x for x in Receiver.received if x['payload']['messageType']=='EARLY_EVENT'),None)
 out={'video':str(a.video),'triggerType':a.trigger_type,'fps':a.fps,'resolution':[a.width,a.height],'sourceFramesIn7Seconds':source_index,'transmittedFrames':transmit,'acceptedFrames':accepted,'duplicateResponses':duplicate,'conflict409':conflict,'queue429':full,'captureSeconds':round(capture_end-start,3),'postCompleteDrainSeconds':round(done-capture_end,3),'finalSeconds':round(done-start,3),'status':status,'earlyEvent':None if not early else early['payload'],'captureStartToEarlyReceiverSeconds':None if not early else round(early['receivedPerf']-start,3),'final':result,'scope':'Streaming API Phase B; results depend on triggerType and contiguous source frames'}
 path=a.output_dir;path.mkdir(parents=True,exist_ok=True);(path/f'{a.trigger_type.lower()}_{a.video.stem}_{int(a.fps)}_7s.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
