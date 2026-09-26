"""Signals over the whole clip, drawn so they can be judged by eye."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import probe_common

d = probe_common.load()
p = np.load("probe2.npz")
t = p["time"]; f = d["feats"]
valid = f["valid"].astype(bool)

def sg(x, w=9, o=2):
    from scipy.signal import savgol_filter
    y = x.copy(); good = np.isfinite(y)
    if good.sum() < w: return y
    y[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), y[good])
    return savgol_filter(y, w, o)

fig, ax = plt.subplots(5, 1, figsize=(16, 13), sharex=True)

ax[0].plot(t, p["omega"], lw=0.8, alpha=.45, color="tab:blue", label="flow omega, raw")
ax[0].plot(t, sg(p["omega"]), lw=2, color="tab:blue", label="flow omega, smoothed")
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_ylabel("rad/s"); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("FLOW: angular rate about the pivot (whole mask) -- sign = swing direction")

ax[1].plot(t, f["slew_rate"], lw=.9, color="tab:red", label="chain slew rate (current pipeline)")
ax[1].axhline(0, color="k", lw=.8); ax[1].legend(loc="upper right", fontsize=8)
ax[1].set_ylabel("rad/s"); ax[1].set_title("CHAIN: the same quantity, from the fitted arm pose")

ax[2].plot(t, p["v_up"], lw=.8, alpha=.45, color="tab:green")
ax[2].plot(t, sg(p["v_up"]), lw=2, color="tab:green", label="flow vertical rate, distal fifth")
ax[2].axhline(0, color="k", lw=.8); ax[2].legend(loc="upper right", fontsize=8)
ax[2].set_ylabel("L/s"); ax[2].set_title("FLOW: is the far end of the arm going up or down?")

ax[3].plot(t, p["speed"], lw=.8, alpha=.45, color="tab:purple")
ax[3].plot(t, sg(p["speed"]), lw=2, color="tab:purple", label="flow speed (median |F| in mask)")
ax[3].legend(loc="upper right", fontsize=8); ax[3].set_ylabel("L/s")
ax[3].set_title("FLOW: how much is the machine moving at all? -- the dwell gate")

ax[4].plot(t, f["elevation"], lw=1.2, color="tab:orange", label="chain elevation")
ax[4].axhline(d["scene"]["surface_height"], color="brown", ls="--", lw=1, label="surface height")
ax[4].legend(loc="upper right", fontsize=8); ax[4].set_ylabel("L")
ax[4].set_xlabel("time (s)"); ax[4].set_title("CHAIN: elevation, for reference")

for a in ax: a.grid(alpha=.25); a.set_xlim(t[0], t[-1])
plt.tight_layout(); plt.savefig("probe_signals.png", dpi=105)
print("wrote probe_signals.png")

# where does flow omega change sign, with any persistence?
w = sg(p["omega"]); good = np.isfinite(w)
sign = np.sign(w)
flips = [i for i in range(1, len(w)) if sign[i] != 0 and sign[i-1] != 0 and sign[i] != sign[i-1]]
# keep only flips where the signal is substantial on both sides for >=0.4 s
strong = []
for i in flips:
    lo, hi = max(0, i-4), min(len(w), i+5)
    if np.max(np.abs(w[lo:i])) > 0.02 and np.max(np.abs(w[i:hi])) > 0.02:
        strong.append(t[i])
print("flow omega sign flips (persistent):", np.round(strong, 2))
