"""Does a bucket-vs-forearm differential show the uncurl, using ONLY the radial
bands we already have? If yes, the idea works before we build a bucket mask."""
import numpy as np, json, cv2
from excavator_cycles.masks import decode
from excavator_cycles.motion import dense_flow
from excavator_cycles.config import Config
from excavator_cycles.video import iter_samples

d=np.load('outputs/track/real/masks.npz'); shape=tuple(d['shape'])
sc=json.load(open('outputs/track/real/scene.json')); piv=np.array(sc['centre'])
keys=sorted([k for k in d.files if k.startswith('m') and k[1:].isdigit()], key=lambda s:int(s[1:]))
masks=[decode(d[k],shape) for k in keys]

cfg=Config()
frames=[s.image for s in iter_samples('construction_excavator_cycle_duration_1.mp4', cfg.sampling.rate_hz)]
assert len(frames)==len(masks), (len(frames),len(masks))
ys,xs=np.mgrid[0:shape[0],0:shape[1]]
rad=np.hypot(xs-piv[0],ys-piv[1])

rows=[]
for i in range(len(frames)-1):
    m=masks[i]
    if m.sum()<200: rows.append((np.nan,)*3); continue
    rmax=rad[m].max()
    tip  = m & (rad>=0.82*rmax)          # outermost ~ the bucket
    fore = m & (rad>=0.50*rmax) & (rad<0.75*rmax)   # the stick/forearm
    if tip.sum()<80 or fore.sum()<80: rows.append((np.nan,)*3); continue
    F=dense_flow(frames[i],frames[i+1])
    fx,fy=F[...,0],F[...,1]
    # median flow of each band
    tvx,tvy=np.median(fx[tip]),np.median(fy[tip])
    fvx,fvy=np.median(fx[fore]),np.median(fy[fore])
    rows.append((tvx-fvx, tvy-fvy, np.hypot(tvx-fvx,tvy-fvy)))
a=np.array(rows)
t=np.arange(len(a))*0.1
np.save('/private/tmp/claude-501/-Users-aanyashah-Desktop-Agent/e9f9cce6-9ab1-42d9-ac3a-15dc4fa46b0f/scratchpad/vrel.npy', np.c_[t,a])
mag=a[:,2]
print("v_rel magnitude (px per 0.1s), band-differential, NO bucket mask\n")
ph={'dig 3-10.5':(3,10.5),'haul 10.5-14':(10.5,14),'DUMP PLATEAU 14.2-25.9':(14.2,25.9),'swing 26-27.3':(26,27.3)}
for k,(lo,hi) in ph.items():
    s=mag[(t>=lo)&(t<hi)]
    print(f"  {k:24s} med={np.nanmedian(s):6.3f}  p90={np.nanpercentile(s,90):6.3f}  max={np.nanmax(s):6.3f}")
print("\nTop 10 v_rel spikes inside the dump plateau (candidate tipping moments):")
sel=np.where((t>=14.2)&(t<25.9))[0]
for i in sel[np.argsort(-mag[sel])][:10]:
    print(f"   t={t[i]:5.1f}s  |v_rel|={mag[i]:5.3f}  (dx={a[i,0]:+.3f}, dy={a[i,1]:+.3f})")
