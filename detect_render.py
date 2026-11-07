from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def draw(frame, detections, color, labels):
 for x1,y1,x2,y2,score,tid in detections:
  x1,y1,x2,y2=map(int,(x1,y1,x2,y2));cv2.rectangle(frame,(x1,y1),(x2,y2),color,max(2,frame.shape[1]//1000))
  text=f'ID {tid} {score:.2f}' if labels else f'{score:.2f}';cv2.putText(frame,text,(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,1.0,color,2,cv2.LINE_AA)

def iou(a,b):
 ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b;ix1=max(ax1,bx1);iy1=max(ay1,by1);ix2=min(ax2,bx2);iy2=min(ay2,by2); inter=max(0,ix2-ix1)*max(0,iy2-iy1); area1=max(1,ax2-ax1)*max(1,ay2-ay1);area2=max(1,bx2-bx1)*max(1,by2-by1);return inter/(area1+area2-inter)

def run(source,out,model,step,conf,imgsz,device,nms_iou):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS) or 25;w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));codec=cv2.VideoWriter_fourcc(*'mp4v');raw_writer=cv2.VideoWriter(str(out/'yolo_only.mp4'),codec,fps,(w,h));track_writer=cv2.VideoWriter(str(out/'tracked.mp4'),codec,fps,(w,h));net=YOLO(model);active={};next_id=1;frame_no=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  current=[]
  if frame_no%step==0:
   result=net(frame,classes=[0],conf=conf,iou=nms_iou,imgsz=imgsz,device=device,verbose=False)[0]
   for box in result.boxes:
    x1,y1,x2,y2=map(int,box.xyxy[0].tolist());score=float(box.conf[0]);current.append((x1,y1,x2,y2,score,None))
   used=set();new={}
   for x1,y1,x2,y2,score,_ in current:
    best=None;best_iou=0
    for tid,(old_box,last) in active.items():
     if tid in used or frame_no-last>step*4:continue
     value=iou((x1,y1,x2,y2),old_box)
     if value>best_iou:best_iou=value;best=tid
    if best is None:best=next_id;next_id+=1
    used.add(best);new[best]=((x1,y1,x2,y2),frame_no);current[current.index((x1,y1,x2,y2,score,None))]=(x1,y1,x2,y2,score,best)
   active={**{k:v for k,v in active.items() if frame_no-v[1]<=step*4},**new}
  raw_frame=frame.copy();tracked_frame=frame.copy();draw(raw_frame,current,(0,255,255),False);draw(tracked_frame,current,(0,220,80),True)
  raw_writer.write(raw_frame), track_writer.write(tracked_frame)
  frame_no += 1
  if frame_no % 20 == 0:
   print(f"Обработано: {frame_no}/{n} кадров ({frame_no/n*100:.1f}%)", flush=True)
 cap.release();raw_writer.release();track_writer.release();print(f'processed {n} frames -> {out}',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',default='videos_proms/camera_new_4 - Trim.mp4');p.add_argument('--output',default='videos_proms_results');p.add_argument('--model',default='yolo11m.pt');p.add_argument('--step',type=int,default=2);p.add_argument('--conf',type=float,default=.15);p.add_argument('--imgsz',type=int,default=1280);p.add_argument('--device',default='cpu');p.add_argument('--nms-iou',type=float,default=.4);a=p.parse_args();run(a.source,a.output,a.model,a.step,a.conf,a.imgsz,a.device,a.nms_iou)

