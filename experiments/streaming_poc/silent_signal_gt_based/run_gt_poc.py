"""GT-guided, causal ownership experiments. Does not change production code."""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import cv2, mediapipe as mp, pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'hand_motion_desktop/src')]
from detector import _build_detector,MIN_FIST_FRAMES,REQUIRED_REPEATS,MAX_GAP_BETWEEN_REPS_FRAMES
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES
from gesture_rules import RepeatedFistToOpenTracker,frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front
from tracking.keypad_rules_front import identify_front_operator

OUT=ROOT/'experiment_results/streaming_poc/silent_signal_gt_based';RUNS=ROOT/'final_demo/results_front_rules/runs/full64_20261020/videos'
CASES={'negative_174439':('03_keypad/static_two_people/20261019_174439.mp4','NEGATIVE'),'negative_174454':('03_keypad/static_two_people/20261019_174454.mp4','NEGATIVE'),'positive_175021':('03_keypad/approach_signal/20261019_175021.mp4','POSITIVE'),'positive_175204':('03_keypad/approach_signal/20261019_175204.mp4','POSITIVE'),'candidate_174955':('03_keypad/approach_signal/20261019_174955.mp4','UNCONFIRMED'),'candidate_175132':('03_keypad/approach_signal/20261019_175132.mp4','UNCONFIRMED'),'candidate_175228':('03_keypad/approach_signal/20261019_175228.mp4','UNCONFIRMED')}

@dataclass
class Obs: x:float;y:float;handedness:str;gesture:str;owner:str;span:float
@dataclass
class Track: ident:int;x:float;y:float;handedness:str;tracker:RepeatedFistToOpenTracker;last_operator:int|None=None;missing:int=0

def stored(relative):
    for path in RUNS.glob('*.json'):
        p=json.loads(path.read_text(encoding='utf8'))
        if p.get('relative_path')==relative:return p
    raise FileNotFoundError(relative)

def owner(ob,boxes,operator,w,h):
    hits=[];x,y=ob.x*w,ob.y*h
    for b in boxes:
        bw=max(float(b['x2'])-float(b['x1']),1);bh=max(float(b['y2'])-float(b['y1']),1);mx,my=max(12,bw*.05),max(12,bh*.05)
        if float(b['x1'])-mx<=x<=float(b['x2'])+mx and float(b['y1'])-my<=y<=float(b['y2'])+my:hits.append(int(b['track_id']))
    if not hits:return 'unassigned'
    if operator in hits and len(hits)==1:return 'operator'
    if operator in hits:return 'ambiguous'
    return 'nonoperator'

class OwnerHandTracker:
    """Per-hand legacy-repeat tracker plus causal ownership admission policies."""
    def __init__(self,policy):self.policy=policy;self.tracks={};self.next=1
    def update(self,observations,frame):
        pairs=[]
        for oi,o in enumerate(observations):
            for tid,t in self.tracks.items():
                d=math.hypot(o.x-t.x,o.y-t.y)
                if d<=.20:pairs.append((d+(0 if o.handedness==t.handedness else .03),tid,oi))
        pairs.sort();ass={};usedt=set();usedo=set()
        for _,tid,oi in pairs:
            if tid not in usedt and oi not in usedo:ass[oi]=tid;usedt.add(tid);usedo.add(oi)
        for oi,o in enumerate(observations):
            if oi not in ass:
                tid=self.next;self.next+=1;self.tracks[tid]=Track(tid,o.x,o.y,o.handedness,RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES));ass[oi]=tid
        triggered=[];details=[];live=set(ass.values())
        for oi,o in enumerate(observations):
            tid=ass[oi];t=self.tracks[tid];t.x,t.y,t.handedness,t.missing=o.x,o.y,o.handedness,0
            if o.owner=='operator':t.last_operator=frame
            # O1 rejects only an explicitly non-operator bbox assignment.
            # O2 additionally requires an operator observation on THIS causal
            # hand track within 30 frames; unknown is never labelled operator.
            allow=o.owner!='nonoperator' if self.policy=='reject_nonoperator' else (o.owner=='operator' or (o.owner in ('ambiguous','unassigned') and t.last_operator is not None and frame-t.last_operator<=30))
            if allow and t.tracker.update(o.gesture):triggered.append(tid)
            elif not allow:t.tracker.update('none')
            details.append({'track':tid,'owner':o.owner,'allowed':allow,'lastOperatorFrame':t.last_operator})
        for tid in list(self.tracks):
            if tid not in live:
                t=self.tracks[tid];t.missing+=1;t.tracker.update('none')
                if t.missing>8:del self.tracks[tid]
        return triggered,details

def run(case,mode):
    relative,gt=CASES[case];video=ROOT/'final_demo/videos'/relative;p=stored(relative);records=p['full']['tracking'];df=pd.DataFrame(records);fps=float(p['fps']);operator=identify_front_operator(df,fps);boxes=defaultdict(list)
    for r in records:boxes[int(r['frame'])].append(r)
    cap=cv2.VideoCapture(str(video));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));detector=_build_detector();baseline=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES);o1=OwnerHandTracker('reject_nonoperator');o2=OwnerHandTracker('recent_operator');hits={'baseline':[],'O1_rejectNonoperator_sameHand':[],'O2_recentOperator_sameHand':[]};trace=[];i=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        work=frame
        if mode=='jpeg':
            work=cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA);okjpg,j=cv2.imencode('.jpg',work,[cv2.IMWRITE_JPEG_QUALITY,85]);work=cv2.imdecode(j,cv2.IMREAD_COLOR)
        rgb=cv2.cvtColor(cv2.flip(work,1),cv2.COLOR_BGR2RGB);r=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(i*1000/fps));obs=[]
        for lm,hand in zip(r.hand_landmarks,r.handedness):
            label=hand[0].category_name;g=classify_gesture_front(lm,label);pts=[lm[k] for k in (0,5,9,17)];x=1-sum(q.x for q in pts)/4;y=sum(q.y for q in pts)/4;span=math.hypot(max(q.x for q in lm)-min(q.x for q in lm),max(q.y for q in lm)-min(q.y for q in lm));temp=Obs(x,y,label,g,'',span);obs.append(Obs(x,y,label,g,owner(temp,boxes[i],operator,w,h),span))
        b=baseline.update(frame_gesture_from_hands(o.gesture for o in obs));a1,d1=o1.update(obs,i);a2,d2=o2.update(obs,i)
        if b:hits['baseline'].append(i)
        if a1:hits['O1_rejectNonoperator_sameHand'].append(i)
        if a2:hits['O2_recentOperator_sameHand'].append(i)
        trace.append({'frame':i,'hands':[{'gesture':o.gesture,'handedness':o.handedness,'centerOriginal':[round(o.x,4),round(o.y,4)],'landmarkSpanNormalized':round(o.span,4),'owner':o.owner,'O1':d1[j],'O2':d2[j]} for j,o in enumerate(obs)]})
        i+=1
    cap.release();detector.close();return {'case':case,'gt':gt,'inputMode':mode,'operatorTrackId':operator,'frames':i,'hits':hits,'trace':trace}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=CASES,required=True);ap.add_argument('--mode',choices=['raw','jpeg'],required=True);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);result=run(a.case,a.mode);path=OUT/f'{a.case}_{a.mode}.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='trace'},ensure_ascii=False))
if __name__=='__main__':main()
