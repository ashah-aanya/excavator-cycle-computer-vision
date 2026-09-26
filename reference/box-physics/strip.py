"""Zoom on one motion I can name from the pictures: the return swing, 24-29s.
Draw the flow and the measured numbers on each frame, so the SIGN can be checked
against motion that is visible rather than assumed."""
import numpy as np, cv2, probe_common, sys
d = probe_common.load()
p = np.load("probe2.npz")
fi, t, col, masks, shape = d["frame_index"], d["times"], d["colour"], d["mask_list"], d["shape"]
centre = np.array(d["scene"]["centre"])
lo, hi, step = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
picks = [int(np.argmin(np.abs(t - s))) for s in np.arange(lo, hi+0.001, step)]
tiles = []
for k in picks:
    a = fi[k]
    if a not in col: continue
    img = cv2.resize(col[a], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    m = masks[k]
    cnts,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, [c*3 for c in cnts], -1, (0,255,255), 1)
    cv2.circle(img, (int(centre[0]*3), int(centre[1]*3)), 5, (255,0,255), -1)
    if k+1 < len(fi) and fi[k+1] in d["grays"] and a in d["grays"]:
        fl = cv2.calcOpticalFlowFarneback(d["grays"][a], d["grays"][fi[k+1]], None,0.5,4,15,3,5,1.2,0)
        for y in range(0, shape[0], 7):
            for x in range(0, shape[1], 7):
                if not m[y,x]: continue
                fx, fy = fl[y,x]
                if np.hypot(fx,fy) < 0.25: continue
                cv2.arrowedLine(img,(x*3,y*3),(int((x+fx*8)*3),int((y+fy*8)*3)),(0,255,0),1,tipLength=.3)
    cv2.rectangle(img,(0,0),(img.shape[1],32),(0,0,0),-1)
    cv2.putText(img,f"t={t[k]:.2f}s  w={p['omega'][k]:+.3f}rad/s",(4,13),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
    cv2.putText(img,f"vup={p['v_up'][k]:+.3f}  spd={p['speed'][k]:.3f} L/s",(4,27),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,255,180),1)
    tiles.append(img)
cols = 4
rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles)-cols+1, cols)]
sheet = np.vstack(rows)
sheet = cv2.resize(sheet,(1700,int(1700*sheet.shape[0]/sheet.shape[1])))
name = f"strip_{lo:.0f}_{hi:.0f}.jpg"
cv2.imwrite(name, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]); print("wrote", name, len(tiles))
