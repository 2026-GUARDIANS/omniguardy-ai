"""Read-only adapters from JPEG streaming frames to existing final rules."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import cv2, numpy as np, pandas as pd

from exhibition_audio_service import (
    EDGE_DENSITY_MAX, LAPLACIAN_VARIANCE_MAX, OCCLUSION_MIN_SUSTAINED_SECONDS,
    ANALYSIS_SECONDS, detect_person_dwell,
)
from tracking.keypad_rules_front import detect_front_rear_approach

TRACKING_COLUMNS=["frame","track_id","x1","y1","x2","y2","confidence"]

class OnlineOcclusionAdapter:
    """Same metric/threshold calculation as detect_camera_occlusion, on frames."""
    def __init__(self): self.rows: list[tuple[int,bool,float,float]]=[]
    def observe(self, source_frame: int, image: np.ndarray) -> None:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edge=float(cv2.countNonZero(cv2.Canny(gray,100,200)))/float(gray.size)
        lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        self.rows.append((source_frame,edge<EDGE_DENSITY_MAX and lap<LAPLACIAN_VARIANCE_MAX,edge,lap))
    def final_events(self,fps:float)->list[dict[str,Any]]:
        required=max(1,round(OCCLUSION_MIN_SUSTAINED_SECONDS*fps));events=[];start=None
        for pos,(_,flag,_,_) in enumerate([*self.rows,(None,False,0.,0.)]):
            if flag and start is None:start=pos
            elif not flag and start is not None:
                if pos-start>=required:
                    first=self.rows[start][0];last=self.rows[pos-1][0]
                    events.append({"eventType":"CAMERA_OCCLUSION_SUSPECTED","confidence":None,"detectedFrame":self.rows[start+required-1][0],"details":{"startSeconds":round(first/fps,3),"endSeconds":round(last/fps,3),"durationSeconds":round((last-first+1)/fps,3),"edgeDensityThreshold":EDGE_DENSITY_MAX,"laplacianThreshold":LAPLACIAN_VARIANCE_MAX,"minSustainedSeconds":OCCLUSION_MIN_SUSTAINED_SECONDS,"windowSeconds":round(len(self.rows)/fps,3),"experimental":True}})
                start=None
        return events

def tracking_dataframe(rows:list[dict[str,Any]],width:int|None,height:int|None)->pd.DataFrame:
    df=pd.DataFrame(rows,columns=TRACKING_COLUMNS)
    if width:df.attrs['frame_width']=width
    if height:df.attrs['frame_height']=height
    return df

def source_is_contiguous(indices:list[int])->bool:
    return bool(indices) and all(b==a+1 for a,b in zip(indices,indices[1:]))

def final_keypad_events(rows:list[dict[str,Any]],fps:float,width:int|None,height:int|None,contiguous:bool)->tuple[list[dict[str,Any]],int|None]:
    df=tracking_dataframe(rows,width,height);count=0 if df.empty else int(df.track_id.nunique())
    if not contiguous:return [],count
    event=detect_front_rear_approach(df,fps)
    return ([] if event is None else [event]),count

def final_audio_events(rows:list[dict[str,Any]],occlusion:OnlineOcclusionAdapter,fps:float,width:int|None,height:int|None,contiguous:bool)->tuple[list[dict[str,Any]],int|None]:
    df=tracking_dataframe(rows,width,height);count=0 if df.empty else int(df.track_id.nunique())
    if not contiguous:return [],count
    window=min(ANALYSIS_SECONDS,max(0.,(max([r['frame'] for r in rows],default=-1)+1)/fps))
    dwell=detect_person_dwell(df,fps,window_seconds=window)
    return [*occlusion.final_events(fps),*dwell],count
