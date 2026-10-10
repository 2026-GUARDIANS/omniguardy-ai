"""Draw keypad ROI plus MediaPipe left/right wrist position and visibility."""
import argparse
from pathlib import Path
import cv2, mediapipe as mp
from mediapipe.tasks.python import vision

def main():
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--roi',required=True,help='x1,y1,x2,y2 normalized');p.add_argument('--frame',type=int,default=0);p.add_argument('--out',type=Path,required=True);p.add_argument('--pose-model',type=Path,default=Path('models/pose_landmarker_lite.task'));a=p.parse_args()
 vals=[float(v) for v in a.roi.split(',')]
 if len(vals)!=4 or not all(0<=v<=1 for v in vals):raise SystemExit('ROI must be x1,y1,x2,y2 in [0,1]')
 cap=cv2.VideoCapture(str(a.input));fps=cap.get(cv2.CAP_PROP_FPS) or 30.;cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame);ok,img=cap.read();cap.release()
 if not ok:raise SystemExit('cannot read requested frame')
 h,w=img.shape[:2];x1,y1,x2,y2=vals;cv2.rectangle(img,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),(0,255,255),2)
 opts=vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(a.pose_model)),running_mode=vision.RunningMode.VIDEO,num_poses=2)
 with vision.PoseLandmarker.create_from_options(opts) as landmarker:
  result=landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),int(a.frame/fps*1000))
 for person,landmarks in enumerate(result.pose_landmarks):
  for label,index,color in [('L',15,(255,0,255)),('R',16,(0,0,255))]:
   q=landmarks[index];xy=(int(q.x*w),int(q.y*h));vis=float(getattr(q,'visibility',0));cv2.circle(img,xy,6,color,-1);cv2.putText(img,f'{person}{label} v={vis:.2f}',(xy[0]+6,xy[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
 a.out.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(a.out),img)
if __name__=='__main__':main()
