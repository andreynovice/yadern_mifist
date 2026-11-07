from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2

def render(videos,report,out,hold):
 data=json.loads(Path(report).read_text(encoding='utf-8')); out=Path(out);out.mkdir(parents=True,exist_ok=True); files=sorted(Path(videos).glob('*.mp4'))
 for camera,src in enumerate(files,1):
  cap=cv2.VideoCapture(str(src));fps=cap.get(cv2.CAP_PROP_FPS) or 25;width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fourcc=cv2.VideoWriter_fourcc(*'mp4v');writer=cv2.VideoWriter(str(out/f'{src.stem}_tracked.mp4'),fourcc,fps,(width,height)); boxes=[]
  for track in data['tracks']:
   for item in track.get('boxes',[]):
    if item['camera']==camera:boxes.append((item['frame'],item['box'],track['global_id']))
  boxes.sort();index=0;active=[];frame=0
  while True:
   ok,image=cap.read()
   if not ok:break
   while index<len(boxes) and boxes[index][0]<=frame:
    f,box,gid=boxes[index];active.append((f,box,gid));index+=1
   kept=[]
   for seen,box,gid in active:
    if frame-seen>hold:continue
    kept.append((seen,box,gid))
    x1,y1,x2,y2=map(int,box);cv2.rectangle(image,(x1,y1),(x2,y2),(0,220,80),max(2,width//1000));label=f'ID {gid}';(tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,1.0,2);cv2.rectangle(image,(x1,max(0,y1-th-12)),(x1+tw+10,y1),(0,220,80),-1);cv2.putText(image,label,(x1+5,max(th+2,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),2,cv2.LINE_AA)
   active=kept
   writer.write(image);frame+=1
  cap.release();writer.release();print(out/f'{src.stem}_tracked.mp4',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--videos',default='videos');p.add_argument('--report',default='outputs_reid/report.json');p.add_argument('--output',default='rendered');p.add_argument('--hold',type=int,default=30);a=p.parse_args();render(a.videos,a.report,a.output,a.hold)

