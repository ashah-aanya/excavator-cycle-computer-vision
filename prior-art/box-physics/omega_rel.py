"""omega_bucket - omega_stick: the bucket JOINT rate, radius-normalised.
Rigid arm -> 0 at any swing speed. Only articulation survives."""
import numpy as np, json
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
assert len(frames)==len(masks)
ys,xs=np.mgrid[0:shape[0],0:shape[1]]
rx,ry=xs-piv[0],ys-piv[1]; rad=np.hypot(rx,ry); rad2=np.maximum(rad**2,1.0)

def omega_of(F,band):
    fx,fy=F[...,0],F[...,1]
    w=(rx[band]*fy[band]-ry[band]*fx[band])/rad2[band]
    return np.median(w)/0.1   # rad/s

out=[]
for i in range(len(frames)-1):
    m=masks[i]
    if m.sum()<200: out.append((np.nan,np.nan,np.nan)); continue
    rmax=rad[m].max()
    buck = m & (rad>=0.82*rmax)
    stick= m & (rad>=0.50*rmax) & (rad<0.75*rmax)
    if buck.sum()<80 or stick.sum()<80: out.append((np.nan,np.nan,np.nan)); continue
    F=dense_flow(frames[i],frames[i+1])
    wb,ws=omega_of(F,buck),omega_of(F,stick)
    out.append((wb,ws,wb-ws))
a=np.array(out); t=np.arange(len(a))*0.1
np.save('/private/tmp/claude-501/-Users-aanyashah-Desktop-Agent/e9f9cce6-9ab1-42d9-ac3a-15dc4fa46b0f/scratchpad/omega_rel.npy',np.c_[t,a])
dw=a[:,2]
print("omega_bucket - omega_stick  (rad/s).  Rigid arm => 0 regardless of swing speed.\n")
print(f"  {'phase':26s} {'w_bucket':>9s} {'w_stick':>9s} {'|DIFF| med':>10s} {'|DIFF| p90':>10s}")
ph={'dig 3-10.5':(3,10.5),'haul 10.5-14':(10.5,14),'DUMP PLATEAU 14.2-25.9':(14.2,25.9),'SWING 26-27.3':(26,27.3)}
for k,(lo,hi) in ph.items():
    s=(t>=lo)&(t<hi)
    print(f"  {k:26s} {np.nanmedian(a[s,0]):+9.3f} {np.nanmedian(a[s,1]):+9.3f} {np.nanmedian(np.abs(dw[s])):10.3f} {np.nanpercentile(np.abs(dw[s]),90):10.3f}")
print("\n  >>> the SWING test: if |diff| during the swing is NOT huge, the confound is gone.")
print("\nTop 8 |diff| spikes inside the dump plateau (candidate tipping):")
sel=np.where((t>=14.2)&(t<25.9))[0]
for i in sel[np.argsort(-np.abs(dw[sel]))][:8]:
    print(f"   t={t[i]:5.1f}s   diff={dw[i]:+6.3f}   (w_bucket={a[i,0]:+.3f}  w_stick={a[i,1]:+.3f})")
