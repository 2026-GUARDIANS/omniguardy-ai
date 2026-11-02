"""Local-only HTTP smoke test for the standalone streaming API PoC."""
import argparse, json, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2, requests

ROOT=Path(__file__).resolve().parents[3]
VIDEO=ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4';received=[]
class Receiver(BaseHTTPRequestHandler):
 def do_POST(self):
  received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))));self.send_response(204);self.end_headers()
 def log_message(self,*x):pass
ap=argparse.ArgumentParser();ap.add_argument('--base',default='http://127.0.0.1:8011');ap.add_argument('--video',type=Path,default=VIDEO);a=ap.parse_args();BASE=a.base
server=ThreadingHTTPServer(('127.0.0.1',0),Receiver);__import__('threading').Thread(target=server.serve_forever,daemon=True).start()
session=f'smoke-{a.video.stem[-6:]}';cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS) or 30.;started=time.perf_counter();r=requests.post(BASE+'/vision/start',json={'sessionId':session,'triggerId':'local-smoke','triggerType':'KEYPAD','sourceFps':fps,'callbackUrl':f'http://127.0.0.1:{server.server_port}/callback','finalResultCallbackEnabled':True},timeout=5);r.raise_for_status()
i=0
while True:
 ok,frame=cap.read()
 if not ok:break
 delay=started+i/fps-time.perf_counter()
 if delay>0:time.sleep(delay)
 small=cv2.resize(frame,(960,540));_,jpg=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,85])
 response=requests.post(BASE+'/vision/frame',data={'sessionId':session,'frameIndex':str(i),'sourceFrameIndex':str(i),'capturedAt':f'{i/fps:.6f}'},files={'frame':('frame.jpg',jpg.tobytes(),'image/jpeg')},timeout=5);response.raise_for_status();i+=1
cap.release();requests.post(BASE+'/vision/complete',json={'sessionId':session},timeout=5).raise_for_status()
for _ in range(80):
 response=requests.get(BASE+f'/vision/result/{session}',timeout=5)
 if response.status_code==200:break
 time.sleep(.25)
result=response.json();server.shutdown();server.server_close();out=ROOT/'experiment_results/streaming_poc/streaming_api_poc/phase_b';out.mkdir(parents=True,exist_ok=True);payload={'video':str(a.video),'sourceFrames':i,'received':received,'final':result,'elapsedSeconds':round(time.perf_counter()-started,3)};(out/f'smoke_{a.video.stem}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(payload,ensure_ascii=False))
