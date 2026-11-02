"""Offline-only experiment for online Silent Signal rule candidates.

This does not modify the legacy detector.  It reuses its MediaPipe model,
mirrored coordinate convention and gesture classification, then feeds the
observations into baseline and experimental online state machines in parallel.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "hand_motion_desktop/src")]
from detector import _build_detector, MIN_FIST_FRAMES, REQUIRED_REPEATS, MAX_GAP_BETWEEN_REPS_FRAMES
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES
from gesture_rules import RepeatedFistToOpenTracker, frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front

OUT = ROOT / "experiment_results/streaming_poc/silent_signal_improvement"
JPEG_QUALITY = 85
VIDEOS = {
    "negative_174439": (ROOT / "final_demo/videos/03_keypad/static_two_people/20261019_174439.mp4", "Human-reviewed negative"),
    "negative_174454": (ROOT / "final_demo/videos/03_keypad/static_two_people/20261019_174454.mp4", "Human-reviewed negative"),
    "candidate_175204": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4", "Candidate; GT unconfirmed"),
    "candidate_174955": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_174955.mp4", "Candidate; GT unconfirmed"),
    "candidate_175021": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_175021.mp4", "Candidate; GT unconfirmed"),
    "candidate_175132": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_175132.mp4", "Candidate; GT unconfirmed"),
    "candidate_175228": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_175228.mp4", "Candidate; GT unconfirmed"),
}


@dataclass
class Obs:
    x: float
    y: float
    handedness: str
    gesture: str
    landmarks: list


@dataclass
class Track:
    ident: int
    x: float
    y: float
    handedness: str
    tracker: RepeatedFistToOpenTracker
    missing: int = 0


class PerHandRepeated:
    """Same-hand version of the legacy *repeated* transition state machine."""

    def __init__(self, radius: float, max_missing: int, stable_open: int = 1):
        self.radius, self.max_missing, self.stable_open = radius, max_missing, stable_open
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.open_streak: dict[int, int] = {}
        self.pending: dict[int, int] = {}

    def update(self, observations: list[Obs]):
        pairs = []
        for oi, ob in enumerate(observations):
            for tid, tr in self.tracks.items():
                d = math.hypot(ob.x - tr.x, ob.y - tr.y)
                if d <= self.radius:
                    pairs.append((d + (0 if ob.handedness == tr.handedness else .03), tid, oi, d))
        pairs.sort()
        used_t, used_o, assignment, distances = set(), set(), {}, {}
        for _, tid, oi, d in pairs:
            if tid not in used_t and oi not in used_o:
                assignment[oi], distances[oi] = tid, d; used_t.add(tid); used_o.add(oi)
        for oi, ob in enumerate(observations):
            if oi not in assignment:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = Track(tid, ob.x, ob.y, ob.handedness, RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS, min_fist_frames=MIN_FIST_FRAMES, max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES, max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES))
                assignment[oi], distances[oi] = tid, None
        triggered = []
        live = set(assignment.values())
        for oi, ob in enumerate(observations):
            tid = assignment[oi]; tr = self.tracks[tid]; tr.x, tr.y, tr.handedness, tr.missing = ob.x, ob.y, ob.handedness, 0
            # Candidate "stable_open_2" only delays an already-complete legacy
            # per-hand sequence by one consecutively observed open-palm frame.
            if ob.gesture == "open_palm": self.open_streak[tid] = self.open_streak.get(tid, 0) + 1
            else: self.open_streak[tid] = 0
            raw = tr.tracker.update(ob.gesture)
            if raw:
                if self.stable_open <= 1 or self.open_streak[tid] >= self.stable_open: triggered.append(tid)
                else: self.pending[tid] = 1
            elif self.pending.get(tid) and ob.gesture == "open_palm":
                self.pending.pop(tid, None); triggered.append(tid)
            elif ob.gesture != "open_palm": self.pending.pop(tid, None)
        for tid in list(self.tracks):
            if tid not in live:
                self.tracks[tid].missing += 1
                # Missing input advances transition time exactly once per frame.
                self.tracks[tid].tracker.update("none")
                if self.tracks[tid].missing > self.max_missing:
                    self.tracks.pop(tid); self.open_streak.pop(tid, None); self.pending.pop(tid, None)
        return triggered, [assignment[i] for i in range(len(observations))], distances


def palm_center(landmarks):
    pts = [landmarks[i] for i in (0, 5, 9, 17)]
    return sum(p.x for p in pts) / 4, sum(p.y for p in pts) / 4


def draw_frame(frame, observations, baseline, strict_ids, robust_ids, index, event_labels):
    canvas = frame.copy(); h, w = canvas.shape[:2]
    for i, ob in enumerate(observations):
        px, py = int((1-ob.x)*w), int(ob.y*h)  # back-map mirrored x for original frame
        cv2.circle(canvas, (px,py), 8, (0,255,0), -1)
        cv2.putText(canvas, f"{ob.gesture} S{strict_ids[i]} R{robust_ids[i]}", (px+8,py-8), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,0), 1)
        for lm in ob.landmarks:
            cv2.circle(canvas, (int((1-lm.x)*w),int(lm.y*h)), 2, (255,180,0), -1)
    cv2.putText(canvas, f"frame {index} baseline={baseline} {' '.join(event_labels)}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,255), 2)
    return canvas


def run_case(name, video, gt):
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector=_build_detector(); base=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES)
    strict=PerHandRepeated(.12,5,1); stable=PerHandRepeated(.12,5,2); robust=PerHandRepeated(.20,8,1)
    rows=[]; hits={"baseline":[],"same_hand_repeated":[],"same_hand_stable_open2":[],"same_hand_tolerant":[]}; index=0
    while True:
        ok, frame=cap.read()
        if not ok: break
        # Match the streaming transport boundary: Vision observes the decoded
        # JPEG, rather than an in-process resize buffer.
        resized=cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA)
        encoded_ok,jpeg=cv2.imencode('.jpg',resized,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
        if not encoded_ok: raise RuntimeError(f'JPEG encode failed at frame {index}')
        decoded=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
        rgb=cv2.cvtColor(cv2.flip(decoded,1),cv2.COLOR_BGR2RGB)
        result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(index*1000/fps))
        obs=[]
        for l, hand in zip(result.hand_landmarks,result.handedness):
            label=hand[0].category_name; g=classify_gesture_front(l,label); x,y=palm_center(l); obs.append(Obs(x,y,label,g,l))
        bg=frame_gesture_from_hands(o.gesture for o in obs); b=base.update(bg)
        sh,si,sd=strict.update(obs); st,sti,std=stable.update(obs); rh,ri,rd=robust.update(obs)
        if b:hits['baseline'].append(index)
        if sh:hits['same_hand_repeated'].append(index)
        if st:hits['same_hand_stable_open2'].append(index)
        if rh:hits['same_hand_tolerant'].append(index)
        rows.append({'frame':index,'seconds':round(index/fps,4),'baselineFrameGesture':bg,'baselineTriggered':b,'hands':[{'gesture':o.gesture,'handedness':o.handedness,'centerMirrored':[round(o.x,4),round(o.y,4)],'strictTrack':si[j],'strictDistance':sd[j],'stableTrack':sti[j],'robustTrack':ri[j],'robustDistance':rd[j]} for j,o in enumerate(obs)],'strictTriggeredTracks':sh,'stableTriggeredTracks':st,'robustTriggeredTracks':rh})
        index+=1
    cap.release(); detector.close()
    # Rerender short source-frame clips for every baseline/candidate event.
    wanted=sorted({f for values in hits.values() for f in values}); clip_info=[]
    for event_frame in wanted:
        cap=cv2.VideoCapture(str(video)); start=max(0,event_frame-20); end=min(index-1,event_frame+20); cap.set(cv2.CAP_PROP_POS_FRAMES,start); writer=None; outpath=OUT/'annotated'/f'{name}_event_{event_frame:04d}.mp4'; outpath.parent.mkdir(parents=True,exist_ok=True)
        for fi in range(start,end+1):
            ok,fr=cap.read()
            if not ok:break
            if writer is None: writer=cv2.VideoWriter(str(outpath),cv2.VideoWriter_fourcc(*'mp4v'),fps,(fr.shape[1],fr.shape[0]))
            r=rows[fi]; # landmarks are not persisted; render centres/text from trace observations.
            cv2.putText(fr, f'frame {fi} baseline={r["baselineFrameGesture"]}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
            cv2.putText(fr, f"strict={r['strictTriggeredTracks']} stable={r['stableTriggeredTracks']} tolerant={r['robustTriggeredTracks']}",(15,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            for hand in r['hands']:
                x=int((1-hand['centerMirrored'][0])*fr.shape[1]); y=int(hand['centerMirrored'][1]*fr.shape[0]); cv2.circle(fr,(x,y),9,(0,255,0),-1); cv2.putText(fr,f"{hand['gesture']} S{hand['strictTrack']} R{hand['robustTrack']}",(x+10,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
            writer.write(fr)
        cap.release()
        if writer:writer.release();clip_info.append(str(outpath.relative_to(ROOT)))
    return {'video':str(video.relative_to(ROOT)),'groundTruth':gt,'fps':fps,'frames':index,'hits':hits,'annotatedClips':clip_info,'trace':str((OUT/'traces'/f'{name}.json').relative_to(ROOT))},rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=VIDEOS.keys());args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    summary=[]
    selected={args.case:VIDEOS[args.case]} if args.case else VIDEOS
    for name,(video,gt) in selected.items():
        case,rows=run_case(name,video,gt);(OUT/'traces'/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=1),encoding='utf-8');summary.append(case);print(name,case['hits'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['video','gt','baseline','same_hand_repeated','same_hand_stable_open2','same_hand_tolerant'])
        for x in summary:w.writerow([x['video'],x['groundTruth'],*[','.join(map(str,x['hits'][k])) for k in ('baseline','same_hand_repeated','same_hand_stable_open2','same_hand_tolerant')]])

if __name__=='__main__':main()
