"""Internal-only keypad hand ROI rule; it never changes API response payloads."""
from __future__ import annotations
from dataclasses import dataclass
import os

EVENT_TYPE = "KEYPAD_REPEATED_HAND_APPROACH"

@dataclass(frozen=True)
class Roi:
    x1: float; y1: float; x2: float; y2: float
    @classmethod
    def parse(cls, raw: str | None):
        if not raw: return None
        try:
            vals=[float(x.strip()) for x in raw.split(',')]
            if len(vals)!=4 or not all(0<=x<=1 for x in vals): return None
            roi=cls(*vals)
            return roi if roi.x1<roi.x2 and roi.y1<roi.y2 else None
        except ValueError: return None
    def inside(self,x,y,margin=0.): return self.x1-margin<=x<=self.x2+margin and self.y1-margin<=y<=self.y2+margin

@dataclass
class HandRuleConfig:
    threshold:int=3; window_seconds:float=10.; exit_margin:float=.03; exit_samples:int=2; missing_samples:int=2; visibility:float=.5
    @classmethod
    def env(cls):
        def get(n,d,cast):
            try:return cast(os.getenv(n,str(d)))
            except ValueError:return d
        return cls(get('VISION_KEYPAD_HAND_ENTRIES',3,int),get('VISION_KEYPAD_HAND_WINDOW_SECONDS',10.,float),get('VISION_KEYPAD_HAND_EXIT_MARGIN',.03,float),get('VISION_KEYPAD_HAND_EXIT_SAMPLES',2,int),get('VISION_KEYPAD_HAND_MISSING_SAMPLES',2,int),get('VISION_KEYPAD_HAND_VISIBILITY',.5,float))

class RepeatedHandApproach:
    """Per-pose-index state. Simultaneous wrists are deliberately one person entry."""
    def __init__(self,roi:Roi,cfg:HandRuleConfig): self.roi,self.cfg,self.state,self.entries=roi,cfg,{},[]
    def observe(self,person:int,frame:int,seconds:float,wrists:list[tuple[float,float,float]]):
        if not 0<=seconds<=self.cfg.window_seconds:return
        st=self.state.setdefault(person,{'inside':False,'outside':0,'missing':0})
        valid=[(x,y) for x,y,v in wrists if v>=self.cfg.visibility]
        if not valid: st['missing']+=1; return
        st['missing']=0; inside=any(self.roi.inside(x,y) for x,y in valid)
        expanded=any(self.roi.inside(x,y,self.cfg.exit_margin) for x,y in valid)
        if not st['inside'] and inside: st.update(inside=True,outside=0); self.entries.append((person,frame,seconds))
        elif st['inside']:
            st['outside']=0 if expanded else st['outside']+1
            if st['outside']>=self.cfg.exit_samples: st.update(inside=False,outside=0)
    def event(self):
        if len(self.entries)<self.cfg.threshold:return None
        _,frame,_=self.entries[self.cfg.threshold-1]
        return {'eventType':EVENT_TYPE,'detectedFrame':frame,'details':{'entryCount':len(self.entries),'threshold':self.cfg.threshold,'windowSeconds':self.cfg.window_seconds}}
