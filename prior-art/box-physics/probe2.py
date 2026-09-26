"""Optical flow probe, correct join. Pass criterion fixed in advance:
   corr(flow omega, chain slew) > 0.8 pass, < 0.5 abandon."""
import numpy as np, cv2, probe_common

d = probe_common.load()
centre = np.array(d["scene"]["centre"]); scale = float(d["scene"]["scale"])
shape = d["shape"]; fi = d["frame_index"]; times = d["times"]
masks = d["mask_list"]; grays = d["grays"]
dt = float(np.median(np.diff(times)))

ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
rx = xs - centre[0]; ry = ys - centre[1]; r2 = rx**2 + ry**2; r = np.sqrt(r2)

n = len(fi)
omega = np.full(n, np.nan); v_rad = np.full(n, np.nan)
v_up = np.full(n, np.nan); speed = np.full(n, np.nan)
omega_arm = np.full(n, np.nan)   # same, but only the DISTAL half of the mask
flows = {}

for k in range(n-1):
    a, b = fi[k], fi[k+1]
    if a not in grays or b not in grays: continue
    m = masks[k]
    if m.sum() < 200: continue
    inner = cv2.erode(m.astype(np.uint8), np.ones((5,5), np.uint8), 1).astype(bool)
    if inner.sum() < 150: inner = m
    flow = cv2.calcOpticalFlowFarneback(grays[a], grays[b], None, 0.5, 4, 15, 3, 5, 1.2, 0)
    flows[k] = flow
    fx, fy = flow[...,0], flow[...,1]
    sel = inner & (r > 1.0)
    if sel.sum() < 100: continue
    w = (ry[sel]*fx[sel] - rx[sel]*fy[sel]) / r2[sel] / dt
    omega[k] = np.median(w)
    v_rad[k] = np.median((rx[sel]*fx[sel] + ry[sel]*fy[sel]) / r[sel]) / scale / dt
    speed[k] = np.median(np.hypot(fx[sel], fy[sel])) / scale / dt
    cut = np.quantile(r[sel], 0.5)
    arm = sel & (r >= cut)
    if arm.sum() > 50:
        omega_arm[k] = np.median((ry[arm]*fx[arm] - rx[arm]*fy[arm]) / r2[arm] / dt)
    cut8 = np.quantile(r[sel], 0.8)
    distal = sel & (r >= cut8)
    if distal.sum() > 30:
        v_up[k] = np.median(-fy[distal]) / scale / dt

np.savez("probe2.npz", time=times, omega=omega, omega_arm=omega_arm,
         v_rad=v_rad, v_up=v_up, speed=speed)

slew = d["feats"]["slew_rate"]; valid = d["feats"]["valid"].astype(bool)
for name, sig in (("whole mask", omega), ("distal half", omega_arm)):
    ok = np.isfinite(sig) & np.isfinite(slew) & valid
    c = np.corrcoef(sig[ok], slew[ok])[0,1]
    mv = ok & (np.abs(slew) > np.quantile(np.abs(slew[ok]), 0.5))
    ag = (np.sign(sig[mv]) == np.sign(slew[mv])).mean()
    print(f"{name:12s} n={ok.sum():3d}  corr={c:+.3f}  sign-agree-while-slewing={ag:.1%}"
          f"  median|w|={np.median(np.abs(sig[ok])):.4f}")
print(f"chain slew    median|w|={np.median(np.abs(slew[valid])):.4f}")

def jit(x):
    dd = np.abs(np.diff(x[np.isfinite(x)])); return np.median(dd), np.quantile(dd,0.95)
print(f"\nframe-to-frame change (median, p95) -- lower is smoother:")
print(f"  flow omega (whole) : {jit(omega)[0]:.4f} {jit(omega)[1]:.4f}")
print(f"  flow omega (distal): {jit(omega_arm)[0]:.4f} {jit(omega_arm)[1]:.4f}")
print(f"  chain slew         : {jit(slew[valid])[0]:.4f} {jit(slew[valid])[1]:.4f}")
