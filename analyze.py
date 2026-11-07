from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import cv2
import numpy as np

class ReID:
 def __init__(self, enabled, device='cpu'):
  self.extractor=None; self.model_name=None
  if not enabled:return
  try:
   from torchreid.reid.utils import FeatureExtractor
   self.model_name='osnet_x1_0'; self.extractor=FeatureExtractor(model_name=self.model_name,device=device)
   print(f'OSNet ReID enabled ({device})',flush=True)
  except Exception as exc:
   raise RuntimeError('OSNet Re-ID requested but could not be initialized. Install requirements.txt and ensure its weights can be downloaded.') from exc
 def vector(self,frame,box):
  if self.extractor is None:return None
  x1,y1,x2,y2=box;crop=frame[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
  if crop.size==0:return None
  value=self.extractor([crop])[0].detach().cpu().numpy().astype(np.float32);return value/(np.linalg.norm(value)+1e-8)

def appearance(frame,box):
 x1,y1,x2,y2=box; crop=frame[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
 if crop.size==0:return np.zeros(48,np.float32)
 hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV); value=cv2.calcHist([hsv],[0,1],None,[8,6],[0,180,0,256]).astype(np.float32)
 return cv2.normalize(value,value).flatten()
def sim(a,b):return float(cv2.compareHist(a.astype(np.float32),b.astype(np.float32),cv2.HISTCMP_CORREL))
class Tracker:
 """Однокамерный Deep SORT. OSNet embedding передаётся в metric Deep SORT."""
 def __init__(self, max_age=50, n_init=3):
  from deep_sort_realtime.deepsort_tracker import DeepSort
  self.tracker=DeepSort(max_age=max_age,n_init=n_init,max_cosine_distance=.35,nn_budget=100,embedder=None)
 def update(self,detections,frame,fps):
  raw=[([d['box'][0],d['box'][1],d['box'][2]-d['box'][0],d['box'][3]-d['box'][1]],d['score'],'person') for d in detections]
  embeds=[d['reid'] if d.get('reid') is not None else d['appearance'] for d in detections]
  tracks=self.tracker.update_tracks(raw,embeds=embeds)
  result=[]
  for track in tracks:
   if not track.is_confirmed() or track.time_since_update>0:continue
   box=tuple(map(int,track.to_ltrb(orig=True)))
   if box[2]<=box[0] or box[3]<=box[1]:continue
   feature=np.asarray(track.get_feature(),dtype=np.float32)
   feature/=np.linalg.norm(feature)+1e-8
   result.append({'box':box,'center':((box[0]+box[2])//2,(box[1]+box[3])//2),'size':(box[2]-box[0],box[3]-box[1]),'local_id':str(track.track_id),'score':track.det_conf or 0.0,'appearance':np.zeros(48,np.float32),'reid':feature})
  return result

def make_detector(model,conf,imgsz):
 if model:
  from ultralytics import YOLO
  net=YOLO(model)
  def detect(frame):
   result=[]
   for b in net(frame,verbose=False,classes=[0],conf=conf,imgsz=imgsz)[0].boxes:
    x1,y1,x2,y2=map(int,b.xyxy[0].tolist());result.append((x1,y1,x2,y2,float(b.conf[0])))
   return result
  return detect,'YOLO11m person detector'
 subtractors={}
 def detect(frame,camera):
  sub=subtractors.setdefault(camera,cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=32,detectShadows=True));mask=sub.apply(frame);mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));mask=cv2.dilate(mask,np.ones((5,5),np.uint8),iterations=2);out=[]
  for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
   x,y,w,h=cv2.boundingRect(contour)
   if w*h>frame.shape[0]*frame.shape[1]*.002 and h>1.25*w and h>45:out.append((x,y,x+w,y+h,.5))
  return out
 return detect,'OpenCV MOG2 fallback'

def process_camera(path,camera,detector,model,step,conf,max_frames,reid):
 cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS) or 25;tracker=Tracker(max_age=max(30,int(fps*2)),n_init=3);segments={};frame=0
 while frame<max_frames if max_frames else True:
  ok,image=cap.read()
  if not ok:break
  if frame%step:frame+=1;continue
  raw=detector(image) if model else detector(image,str(camera))
  detections=[]
  for x1,y1,x2,y2,score in raw:
   box=(x1,y1,x2,y2);detections.append({'box':box,'center':((x1+x2)//2,(y1+y2)//2),'size':(x2-x1,y2-y1),'appearance':appearance(image,box),'reid':reid.vector(image,box),'score':score})
  for item in tracker.update(detections,frame,fps):
   key=item['local_id']; segment=segments.setdefault(key,{'camera':camera,'local_id':key,'start_time':frame/fps,'end_time':frame/fps,'frames':0,'appearances':[],'reids':[],'boxes':[]});segment['end_time']=frame/fps;segment['frames']+=1
   segment['reids'].append(item['reid']);
   if len(segment['boxes'])<300:segment['boxes'].append({'camera':camera,'frame':frame,'box':list(item['box'])})
  frame+=1
 total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
 for segment in segments.values():
  segment['appearance']=np.mean(segment.pop('appearances'),axis=0) if segment['appearances'] else np.zeros(48,np.float32)
  segment['reid']=np.mean(segment.pop('reids'),axis=0) if segment['reids'] else None
  if segment['reid'] is not None:segment['reid']/=np.linalg.norm(segment['reid'])+1e-8
 segments=[s for s in segments.values() if s['frames']>=3]
 return segments,fps,total_frames

def merge_segments(segments,gap):
 parent=list(range(len(segments))); members={i:[i] for i in range(len(segments))}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def compatible(a,b):
  for left in members[a]:
   for right in members[b]:
    x,y=segments[left],segments[right]
    if x['camera']==y['camera'] and min(x['end_time'],y['end_time'])-max(x['start_time'],y['start_time'])>=0:return False
  return True
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b and compatible(a,b):parent[b]=a;members[a].extend(members[b]);del members[b]
 order=sorted(range(len(segments)),key=lambda i:segments[i]['start_time'])
 for pos,i in enumerate(order):
  a=segments[i]
  for j in order[pos+1:]:
   b=segments[j]
   if b['start_time']-a['end_time']>gap:break
   if a['camera']==b['camera']:continue
   overlap=min(a['end_time'],b['end_time'])-max(a['start_time'],b['start_time'])
   temporal=overlap>=0 or abs(a['start_time']-b['end_time'])<=gap or abs(b['start_time']-a['end_time'])<=gap
   reid_score=sim(a['reid'],b['reid']) if a['reid'] is not None and b['reid'] is not None else sim(a['appearance'],b['appearance'])
   if temporal and reid_score>=.58:union(i,j)
 groups={}
 for i,s in enumerate(segments):groups.setdefault(find(i),[]).append(s)
 result=[]
 for gid,items in enumerate(groups.values(),1):
  items=sorted(items,key=lambda x:x['start_time']);cameras=sorted({x['camera'] for x in items});boxes=[]
  for x in items:boxes.extend(x['boxes'])
  result.append({'global_id':gid,'start_time':round(min(x['start_time'] for x in items),2),'end_time':round(max(x['end_time'] for x in items),2),'cameras':cameras,'frames':sum(x['frames'] for x in items),'boxes':boxes[:500]})
 return result

def analyze(videos,out,model=None,step=5,conf=.15,imgsz=1280,gap=8,max_frames=0,reid_enabled=True,device='cpu'):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);files=sorted(Path(videos).glob('*.mp4'))
 if not files:raise FileNotFoundError(f'No MP4 files found in {videos}')
 detector,name=make_detector(model,conf,imgsz);reid_engine=ReID(bool(model) and reid_enabled,device);segments=[];metadata=[]
 for camera,path in enumerate(files,1):
  print(f'camera {camera}/{len(files)}: {path.name}',flush=True);local,fps,total=process_camera(path,camera,detector,model,step,conf,max_frames,reid_engine);segments.extend(local);metadata.append({'camera':camera,'file':path.name,'fps':fps,'frames':total,'local_segments':len(local)})
 tracks=merge_segments(segments,gap)
 report={'detector':name,'reid':reid_engine.model_name,'reid_device':device if reid_engine.extractor else None,'sample_step':step,'association_gap_sec':gap,'cameras':metadata,'total_tracks':len(tracks),'tracks':tracks};(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 with (out/'tracks.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f);writer.writerow(['track_id','start_sec','end_sec','cameras','sampled_frames']);[writer.writerow([t['global_id'],t['start_time'],t['end_time'],','.join(map(str,t['cameras'])),t['frames']]) for t in tracks]
 print(json.dumps({'detector':name,'local_segments':len(segments),'total_tracks':len(tracks),'output':str(out)},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--videos',default='videos');p.add_argument('--output',default='outputs');p.add_argument('--model');p.add_argument('--step',type=int,default=5);p.add_argument('--conf',type=float,default=.15);p.add_argument('--imgsz',type=int,default=1280);p.add_argument('--gap',type=float,default=8);p.add_argument('--max-frames',type=int,default=0);p.add_argument('--device',default='cpu');p.add_argument('--no-reid',action='store_true');a=p.parse_args();analyze(a.videos,a.output,a.model,a.step,a.conf,a.imgsz,a.gap,a.max_frames,not a.no_reid,a.device)





