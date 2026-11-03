"""Local mock integration check for the standalone Streaming API.

It deliberately targets only localhost, simulates Pi multipart uploads, and
never contacts Spring Boot or the port-8000 MP4 API.
"""
from __future__ import annotations
import argparse, json, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2, numpy as np, requests

ROOT=Path(__file__).resolve().parents[3]
VIDEO=ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4'

class Callback(BaseHTTPRequestHandler):
    records=[]; fail_first=True; expected_callback_key=None
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        # Retain only a boolean so generated result files cannot expose a key.
        received_key=self.headers.get('X-Vision-Callback-Key')
        Callback.records.append({'receivedPerf':time.perf_counter(),'payload':body,
                                 'hasVisionCallbackKey': bool(received_key),
                                 'visionCallbackKeyMatchesEnvironment': received_key == Callback.expected_callback_key,
                                 'contentTypeIsJson': self.headers.get_content_type() == 'application/json'})
        if Callback.fail_first:
            Callback.fail_first=False; self.send_response(500)
        else: self.send_response(204)
        self.end_headers()
    def log_message(self,*args): pass

def frame_post(base,sid,index,source,jpeg):
    return requests.post(base+'/vision/frame',data={'sessionId':sid,'frameIndex':str(index),'sourceFrameIndex':str(source),'capturedAt':f'{source/30:.6f}'},files={'frame':('frame.jpg',jpeg,'image/jpeg')},timeout=5)

def wait_result(base,sid,limit=45):
    until=time.monotonic()+limit
    while time.monotonic()<until:
        status=requests.get(base+f'/vision/status/{sid}',timeout=5)
        if status.status_code==200 and status.json()['finalResultAvailable']:
            first=requests.get(base+f'/vision/result/{sid}',timeout=5).json()
            second=requests.get(base+f'/vision/result/{sid}',timeout=5).json()
            return status.json(),first,second
        time.sleep(.1)
    raise TimeoutError(sid)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default='http://127.0.0.1:8010'); ap.add_argument('--output-dir',type=Path,default=ROOT/'experiment_results/streaming_poc/streaming_api_poc/phase_d'); args=ap.parse_args()
    expected_callback_key=__import__('os').environ.get('STREAMING_POC_CALLBACK_KEY')
    if not expected_callback_key:
        raise RuntimeError('STREAMING_POC_CALLBACK_KEY must be set for this callback authentication smoke test')
    Callback.records=[]; Callback.fail_first=True; Callback.expected_callback_key=expected_callback_key
    receiver=ThreadingHTTPServer(('127.0.0.1',0),Callback); threading.Thread(target=receiver.serve_forever,daemon=True).start()
    base=args.base; tiny_ok,tiny=cv2.imencode('.jpg',np.zeros((8,8,3),np.uint8)); assert tiny_ok
    probe=f'contract-probe-{uuid.uuid4().hex[:8]}'; main_id=f'contract-main-{uuid.uuid4().hex[:8]}'
    checks={}
    checks['unknown_status']=requests.get(base+'/vision/status/no-such-session',timeout=5).status_code
    checks['missing_trigger']=requests.post(base+'/vision/start',json={'sessionId':'missing-'+uuid.uuid4().hex[:6],'triggerType':'KEYPAD','sourceFps':30},timeout=5).status_code
    probe_body={'sessionId':probe,'triggerId':'contract-probe-trigger','triggerType':'KEYPAD','sourceFps':30}
    checks['probe_start']=requests.post(base+'/vision/start',json=probe_body,timeout=5).status_code
    checks['duplicate_session']=requests.post(base+'/vision/start',json=probe_body,timeout=5).status_code
    checks['out_of_order']=frame_post(base,probe,1,1,tiny.tobytes()).status_code
    checks['first_frame']=frame_post(base,probe,0,0,tiny.tobytes()).status_code
    duplicate=frame_post(base,probe,0,0,tiny.tobytes()); checks['duplicate_frame']={'http':duplicate.status_code,'body':duplicate.json()}
    checks['complete_first']=requests.post(base+'/vision/complete',json={'sessionId':probe},timeout=5).status_code
    checks['complete_second']=requests.post(base+'/vision/complete',json={'sessionId':probe},timeout=5).status_code
    checks['post_complete_frame']=frame_post(base,probe,1,1,tiny.tobytes()).status_code
    probe_status,probe_first,probe_second=wait_result(base,probe)
    checks['probe_result_idempotent']=probe_first==probe_second

    main_body={'sessionId':main_id,'triggerId':'contract-main-trigger','triggerType':'KEYPAD','sourceFps':30.0,'callbackUrl':f'http://127.0.0.1:{receiver.server_port}/callback'}
    checks['main_start']=requests.post(base+'/vision/start',json=main_body,timeout=5).status_code
    cap=cv2.VideoCapture(str(VIDEO)); started=time.perf_counter(); sent=0
    while sent<210:
        ok,image=cap.read()
        if not ok: break
        deadline=started+sent/30; wait=deadline-time.perf_counter()
        if wait>0: time.sleep(wait)
        image=cv2.resize(image,(960,540)); ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85]); assert ok
        response=frame_post(base,main_id,sent,sent,jpg.tobytes())
        if response.status_code!=202: raise RuntimeError(f'frame {sent}: {response.status_code} {response.text}')
        sent+=1
    cap.release(); capture_end=time.perf_counter()
    checks['main_complete']=requests.post(base+'/vision/complete',json={'sessionId':main_id},timeout=5).status_code
    main_status,main_first,main_second=wait_result(base,main_id); finished=time.perf_counter()
    early=[x for x in Callback.records if x['payload'].get('messageType')=='EARLY_EVENT']
    result_events=main_first['result']['visionEvents']; early_ids=main_first['relatedEarlyEventInstanceIds']
    checks['main_result_idempotent']=main_first==main_second
    checks['callback_attempt_count']=len(Callback.records)
    checks['callback_key_present_on_every_attempt']=bool(Callback.records) and all(x['hasVisionCallbackKey'] for x in Callback.records)
    checks['callback_key_matches_environment_on_every_attempt']=bool(Callback.records) and all(x['visionCallbackKeyMatchesEnvironment'] for x in Callback.records)
    checks['callback_content_type_json_on_every_attempt']=bool(Callback.records) and all(x['contentTypeIsJson'] for x in Callback.records)
    checks['callback_retry_same_instance']=len(early)>=2 and early[0]['payload']['eventInstanceId']==early[1]['payload']['eventInstanceId'] and early[0]['payload']['sequence']==early[1]['payload']['sequence']
    required_early={'sessionId':str,'triggerId':str,'securityEventId':type(None),'triggerType':str,'eventInstanceId':str,'sequence':int,'provisional':bool,'event':dict,'emittedAt':str}
    checks['early_event_contract_types']=bool(early) and all(isinstance(early[-1]['payload'].get(name),kind) for name,kind in required_early.items())
    checks['callback_204_processed']=any(x.get('status') == 204 for x in main_status['callbackAttempts'])
    checks['early_in_final']=bool(early) and any(e.get('eventType')=='SILENT_SIGNAL' and e.get('detectedFrame')==early[-1]['payload']['event']['detectedFrame'] for e in result_events) and early[-1]['payload']['eventInstanceId'] in early_ids
    out={'scope':'localhost mock only; no Pi, Spring Boot, or port-8000 API; callback key values are intentionally excluded','checks':checks,'main':{'framesSent':sent,'captureSeconds':round(capture_end-started,3),'postCompleteDrainSeconds':round(finished-capture_end,3),'finalSeconds':round(finished-started,3),'status':main_status,'final':main_first,'callbackRecords':Callback.records},'probe':{'status':probe_status,'final':probe_first}}
    # First render contains only safe header-presence/match booleans.  Check
    # the actual value before producing the final persisted result.
    checks['callback_key_absent_from_result']=expected_callback_key not in json.dumps(out,ensure_ascii=False)
    if not checks['callback_key_absent_from_result']:
        raise RuntimeError('callback key would be exposed in test output')
    rendered=json.dumps(out,ensure_ascii=False,indent=2)
    args.output_dir.mkdir(parents=True,exist_ok=True); path=args.output_dir/'integration_contract_smoke.json'; path.write_text(rendered,encoding='utf8'); print(rendered); receiver.shutdown(); receiver.server_close()

if __name__=='__main__': main()
