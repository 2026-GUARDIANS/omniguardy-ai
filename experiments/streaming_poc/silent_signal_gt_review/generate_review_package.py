"""Create non-destructive manual GT review assets from existing experiment traces."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import cv2, mediapipe as mp

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'hand_motion_desktop/src')]
from detector import _build_detector
from gesture_rules_front import classify_gesture_front

SRC=ROOT/'experiment_results/streaming_poc/silent_signal_improvement'
OUT=ROOT/'experiment_results/streaming_poc/silent_signal_gt_review'
VIDEOS={
'negative_174439':ROOT/'final_demo/videos/03_keypad/static_two_people/20261019_174439.mp4',
'negative_174454':ROOT/'final_demo/videos/03_keypad/static_two_people/20261019_174454.mp4',
'candidate_175204':ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4',
'candidate_174955':ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_174955.mp4',
'candidate_175021':ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175021.mp4',
'candidate_175132':ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175132.mp4',
'candidate_175228':ROOT/'final_demo/videos/03_keypad/approach_signal/20261019_175228.mp4'}
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

def clusters(frames, gap=120):
    result=[]
    for f in sorted(frames):
        if not result or f-result[-1][-1]>gap: result.append([f])
        else: result[-1].append(f)
    return result

def events_for(case):
    trace=json.loads((SRC/'traces'/f'{case}.json').read_text(encoding='utf8'))
    items=[]
    for row in trace:
        if row['baselineTriggered']:items.append((row['frame'],'baseline'))
        if row['strictTriggeredTracks']:items.append((row['frame'],'same_hand_repeated'))
        if row['stableTriggeredTracks']:items.append((row['frame'],'same_hand_stable_open2'))
        if row['robustTriggeredTracks']:items.append((row['frame'],'same_hand_tolerant'))
    return trace,items

def overlay(frame,row,hand_result):
    h,w=frame.shape[:2]; out=frame.copy()
    for i,(landmarks,handedness) in enumerate(zip(hand_result.hand_landmarks,hand_result.handedness)):
        colour=(0,255,0) if i%2==0 else (0,200,255)
        for a,b in EDGES:
            pa,pb=landmarks[a],landmarks[b]; cv2.line(out,(int((1-pa.x)*w),int(pa.y*h)),(int((1-pb.x)*w),int(pb.y*h)),colour,2)
        for p in landmarks:cv2.circle(out,(int((1-p.x)*w),int(p.y*h)),3,colour,-1)
        gesture=classify_gesture_front(landmarks,handedness[0].category_name)
        info=row['hands'][i] if i<len(row['hands']) else {}
        x,y=int((1-landmarks[0].x)*w),int(landmarks[0].y*h)
        text=f"{handedness[0].category_name} {gesture} S{info.get('strictTrack','?')} R{info.get('robustTrack','?')}"
        cv2.putText(out,text,(x+8,max(25,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,colour,2)
    flags=[]
    if row['baselineTriggered']:flags.append('BASELINE')
    if row['strictTriggeredTracks']:flags.append('SAME_HAND')
    if row['stableTriggeredTracks']:flags.append('STABLE_OPEN2')
    if row['robustTriggeredTracks']:flags.append('TOLERANT')
    cv2.rectangle(out,(0,0),(w,70),(0,0,0),-1)
    cv2.putText(out,f"frame={row['frame']} t={row['seconds']:.3f}s frameGesture={row['baselineFrameGesture']}",(15,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    cv2.putText(out,' | '.join(flags) if flags else 'no event in this frame',(15,57),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255) if flags else (180,180,180),2)
    return out

def montage(frames,path,title):
    thumbw,thumbh=480,270
    if len(frames)>6:
        # Preserve temporal coverage rather than silently making an invalid
        # grid when a cluster contains many closely spaced candidate frames.
        frames=[frames[round(i*(len(frames)-1)/5)] for i in range(6)]
    grid=[]
    for f in frames:
        tile=cv2.resize(f,(thumbw,thumbh));grid.append(tile)
    while len(grid)<6:grid.append(255*cv2.UMat(thumbh,thumbw,cv2.CV_8UC3).get())
    image=cv2.vconcat([cv2.hconcat(grid[:3]),cv2.hconcat(grid[3:])]);cv2.putText(image,title,(15,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2);cv2.imwrite(str(path),image)

def render_case(case):
    trace,events=events_for(case); byframe={x['frame']:x for x in trace}; video=VIDEOS[case]
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30.;total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    detector=_build_detector(); meta=[]
    for ci,frames in enumerate(clusters([f for f,_ in events])):
        start=max(0,min(frames)-int(2*fps));end=min(total-1,max(frames)+int(2*fps));base=f'{case}_frames_{start:04d}_{end:04d}'
        rawpath=OUT/'original'/f'{base}.mp4';anpath=OUT/'analysis'/f'{base}_analysis.mp4';montagepath=OUT/'montage'/f'{base}_montage.jpg'
        for p in (rawpath.parent,anpath.parent,montagepath.parent):p.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start);raw=None;ann=None;selected=[]
        selected_frames=sorted(set([start,end,*frames,*[max(start,f-int(fps*.5)) for f in frames],*[min(end,f+int(fps*.5)) for f in frames]]))
        for index in range(start,end+1):
            ok,frame=cap.read()
            if not ok:break
            if raw is None:
                fourcc=cv2.VideoWriter_fourcc(*'mp4v');raw=cv2.VideoWriter(str(rawpath),fourcc,fps,(frame.shape[1],frame.shape[0]));ann=cv2.VideoWriter(str(anpath),fourcc,fps,(frame.shape[1],frame.shape[0]))
            raw.write(frame)
            resized=cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA);okjpg,jpeg=cv2.imencode('.jpg',resized,[cv2.IMWRITE_JPEG_QUALITY,85]);decoded=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
            rgb=cv2.cvtColor(cv2.flip(decoded,1),cv2.COLOR_BGR2RGB);result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(index*1000/fps))
            rendered=overlay(frame,byframe[index],result);ann.write(rendered)
            if index in selected_frames:selected.append(rendered)
        cap.release();
        if raw:raw.release();ann.release()
        if selected:montage(selected,montagepath,f'{case}: source frames {start}-{end}; events={frames}')
        meta.append({'eventFrames':frames,'sourceStartFrame':start,'sourceEndFrame':end,'originalClipPath':str(rawpath.relative_to(ROOT)),'analysisClipPath':str(anpath.relative_to(ROOT)),'montagePath':str(montagepath.relative_to(ROOT))})
    detector.close();return meta

def make_csv(asset_map):
    rows=[]
    for case in VIDEOS:
        _,events=events_for(case);event_modes={}
        for frame,mode in events:event_modes.setdefault(frame,[]).append(mode)
        for frame,modes in event_modes.items():
            clip=next(x for x in asset_map[case] if frame in x['eventFrames'])
            isneg=case.startswith('negative_')
            rows.append({'videoId':case,'eventFrame':frame,'baselineDetected':'YES' if 'baseline' in modes else 'NO','sameHandDetected':'YES' if any(x!='baseline' for x in modes) else 'NO','humanGT':'NEGATIVE' if isneg else 'UNREVIEWED','sameHandGestureObserved':'UNCERTAIN','observedAction':'','notes':'Video-level human-reviewed negative; do not infer detailed action from this row.' if isneg else 'Existing model candidate only; manual GT required.','reviewClipPath':clip['analysisClipPath']})
    path=OUT/'manual_gt_labels.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['videoId','eventFrame','baselineDetected','sameHandDetected','humanGT','sameHandGestureObserved','observedAction','notes','reviewClipPath']);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=VIDEOS);p.add_argument('--csv-only',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    asset_path=OUT/'asset_manifest.json';assets=json.loads(asset_path.read_text(encoding='utf8')) if asset_path.exists() else {}
    if not a.csv_only:
        targets=[a.case] if a.case else list(VIDEOS)
        for case in targets:assets[case]=render_case(case);print('rendered',case)
        asset_path.write_text(json.dumps(assets,ensure_ascii=False,indent=2),encoding='utf8')
    if len(assets)==len(VIDEOS):make_csv(assets)
if __name__=='__main__':main()
