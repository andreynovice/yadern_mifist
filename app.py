import json, tkinter as tk
from pathlib import Path
import cv2
try:
 from PIL import Image,ImageTk
except ImportError: Image=ImageTk=None
class App:
 def __init__(self,root,video_dir='videos',report='outputs_reid/report.json'):
  self.root=root;self.root.title('Cross-camera tracking');self.r=json.loads(Path(report).read_text(encoding='utf-8'));self.caps=[cv2.VideoCapture(str(p)) for p in sorted(Path(video_dir).glob('*.mp4'))];self.frame=0;self.paused=False;self.labels=[]
  top=tk.Frame(root);top.pack(fill='x');tk.Label(top,text=f"Tracks: {self.r['total_tracks']} | {self.r['detector']}").pack(side='left',padx=10,pady=8);tk.Button(top,text='Play / pause',command=self.toggle).pack(side='right',padx=10);body=tk.Frame(root);body.pack(fill='both',expand=True)
  for i in range(min(4,len(self.caps))):
   body.grid_rowconfigure(i//2,weight=1);body.grid_columnconfigure(i%2,weight=1);lab=tk.Label(body);lab.grid(row=i//2,column=i%2,sticky='nsew',padx=4,pady=4);self.labels.append(lab)
  root.after(40,self.tick)
 def toggle(self):self.paused=not self.paused
 def tick(self):
  if not self.paused:
   for i,cap in enumerate(self.caps[:len(self.labels)]):
    cap.set(cv2.CAP_PROP_POS_FRAMES,self.frame);ok,fr=cap.read()
    if not ok:continue
    fr=cv2.resize(fr,(640,480))
    for track in self.r['tracks']:
     for box in track.get('boxes',[]):
      if box['camera']==i+1 and abs(box['frame']-self.frame)<=12:
       x1,y1,x2,y2=box['box'];scale_x=640/2592;scale_y=480/1944;x1,y1,x2,y2=[int(v) for v in (x1*scale_x,y1*scale_y,x2*scale_x,y2*scale_y)];cv2.rectangle(fr,(x1,y1),(x2,y2),(0,220,80),2);cv2.putText(fr,f"ID {track['global_id']}",(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,80),2)
    if ImageTk:
     photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)));self.labels[i].configure(image=photo);self.labels[i].image=photo
   self.frame+=1
  self.root.after(40,self.tick)
if __name__=='__main__':App(tk.Tk()).root.mainloop()

