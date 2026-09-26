"""Annotated progression video: flow field + the three motion signals, whole clip."""
import numpy as np, cv2, probe_common
d = probe_common.load(); p = np.load("probe2.npz")
fi,t,col,grays,masks,shape = d["frame_index"],d["times"],d["colour"],d["grays"],d["mask_list"],d["shape"]
centre = np.array(d["scene"]["centre"]); n = len(fi)
UP = 3; W, H = shape[1]*UP, shape[0]*UP
STRIP = 150
def sg(x,w=9,o=2):
    from scipy.signal import savgol_filter
    y=x.copy(); g=np.isfinite(y)
    y[~g]=np.interp(np.flatnonzero(~g),np.flatnonzero(g),y[g]); return savgol_filter(y,w,o)
wd, vu, sp = sg(p["omega_arm"]), sg(p["v_up"]), sg(p["speed"])
def norm(x): 
    m=np.nanmax(np.abs(x)); return x/m if m>0 else x
wn, vn, sn = norm(wd), norm(vu), sp/np.nanmax(sp)

vw = cv2.VideoWriter("flow_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H+STRIP))
for k in range(n-1):
    a, b = fi[k], fi[k+1]
    if a not in grays or b not in grays: continue
    img = cv2.resize(col[a], None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    m = masks[k]
    cnts,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,[c*UP for c in cnts],-1,(0,255,255),1)
    cv2.circle(img,(int(centre[0]*UP),int(centre[1]*UP)),5,(255,0,255),-1)
    fl = cv2.calcOpticalFlowFarneback(grays[a],grays[b],None,0.5,4,15,3,5,1.2,0)
    for y in range(0,shape[0],6):
        for x in range(0,shape[1],6):
            if not m[y,x]: continue
            fx,fy = fl[y,x]; mg=np.hypot(fx,fy)
            if mg<0.25: continue
            c = (0,255,0) if mg>0.8 else (0,200,255)
            cv2.arrowedLine(img,(x*UP,y*UP),(int((x+fx*8)*UP),int((y+fy*8)*UP)),c,1,tipLength=.3)
    cv2.rectangle(img,(0,0),(W,46),(0,0,0),-1)
    cv2.putText(img,f"t={t[k]:5.2f}s   swing w={p['omega_arm'][k]:+.3f} rad/s",(6,17),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(img,f"bucket vertical={p['v_up'][k]:+.3f} L/s   speed={p['speed'][k]:.3f} L/s",(6,37),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,255,180),1)
    # signal strip with a playhead
    strip = np.full((STRIP,W,3),25,np.uint8)
    for sig,colr,off,lab in ((wn,(255,160,60),25,"swing w"),(vn,(60,220,60),70,"bucket up/down"),(sn,(200,120,255),115,"speed")):
        pts=[(int(i*W/(n-1)), int(off - sig[i]*18)) for i in range(n-1) if np.isfinite(sig[i])]
        for q in range(1,len(pts)): cv2.line(strip,pts[q-1],pts[q],colr,1)
        cv2.line(strip,(0,off),(W,off),(70,70,70),1)
        cv2.putText(strip,lab,(4,off-22),cv2.FONT_HERSHEY_SIMPLEX,0.35,colr,1)
    x0=int(k*W/(n-1)); cv2.line(strip,(x0,0),(x0,STRIP),(255,255,255),1)
    vw.write(np.vstack([img,strip]))
vw.release(); print("wrote flow_annotated.mp4")
