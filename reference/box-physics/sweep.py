"""Is smoothing necessary, and which smoothing is doing the work?

The flaw in the earlier comparison: the trailing average ran at 0.3-0.5 s while
every derivative went through Savitzky-Golay at 0.9 s. The SG window dominated,
so varying the trailing window barely moved anything -- and "no smoothing at
all" was never run as a control.
"""

import numpy as np

from excavator_cycles.geometry import otsu_threshold
from excavator_cycles.onsets import derivative, motion_boundary

SD = "/private/tmp/claude-501/-Users-aanyashah-Desktop-Agent/e9f9cce6-9ab1-42d9-ac3a-15dc4fa46b0f/scratchpad"
r = np.load(f"{SD}/boxfeat.npz")
piv = r["piv"]; L = float(r["L"]); n = 296
t = np.arange(n) * 0.1
F = 29.97396912419384
TRD = {"dig": (322 - 125) / F, "haul": (559 - 322) / F, "dump": (691 - 559) / F,
       "swing": (880 - 691) / F, "cyc": (880 - 125) / F}
TRT = [125 / F, 322 / F, 559 / F, 691 / F]


def trail(v, w):
    if w <= 1:
        return v.astype(float).copy()
    o = np.full(len(v), np.nan)
    for i in range(len(v)):
        s = v[max(0, i - w + 1): i + 1]
        s = s[np.isfinite(s)]
        if len(s):
            o[i] = s.mean()
    return o


def iv(f, times, ms):
    out, st = [], None
    least = max(1, round(ms / 0.1))
    for i in range(len(f) + 1):
        on = i < len(f) and f[i]
        if on and st is None:
            st = i
        elif not on and st is not None:
            if i - st >= least:
                out.append((times[st], times[i - 1]))
            st = None
    return out


def solve(w, sgw):
    sm = lambda v: trail(v, w)                                  # noqa: E731
    D = lambda v: derivative(v, t, window_seconds=sgw)          # noqa: E731
    X = sm(r["bx"]) / L; Y = sm(r["by"]) / L; OV = sm(r["ov"])
    AR = sm(r["bw"]) / np.maximum(sm(r["bh"]), 1e-9)
    H = (piv[1] / L) - Y
    dH = D(H); ddH = D(dH); SP = np.abs(D(X))
    eps = iv(H < otsu_threshold(H[np.isfinite(H)]), t, 1.0)
    if len(eps) < 2:
        return None
    dig = max(eps, key=lambda a: a[1] - a[0])
    tk = iv(OV > otsu_threshold(OV[np.isfinite(OV)]), t, 0.5)
    if not tk:
        return None
    truck = max(tk, key=lambda a: a[1] - a[0])
    mov = [m for m in iv(SP > otsu_threshold(SP[np.isfinite(SP)]), t, 0.4) if m[0] > truck[0]]
    mm = (t >= dig[0]) & (t <= dig[1])
    low = float(t[np.nanargmin(np.where(mm, H, np.inf))])
    T1 = motion_boundary(dH, t, (0.0, dig[1]), mode="arrives")
    T2 = motion_boundary(ddH, t, (low, truck[0]), mode="arrives")
    T4 = motion_boundary(SP, t, (truck[0], mov[0][1]), mode="departs") if mov else None
    tm = (t >= truck[0]) & (t <= truck[1])
    T3 = float(t[np.nanargmax(np.where(tm, AR, -np.inf))])
    if None in (T1, T2, T4):
        return None
    st = [a for a, _ in eps]
    return T1, T2, T3, T4, len(eps) - 1, float(np.mean(np.diff(st))), st[1] + (T1 - st[0])


def fields(s):
    T1, T2, T3, T4, cc, acd, nxt = s
    d = {"dig": T2 - T1, "haul": T3 - T2, "dump": T4 - T3, "swing": nxt - T4, "cyc": acd}
    return (cc == 1) + sum(1 for k in d if abs(d[k] - TRD[k]) <= 0.6), d


SGS = [0.3, 0.5, 0.7, 0.9, 1.3]
print("GRADED FIELDS PASSING / 6")
print("rows = trailing average window,  cols = Savitzky-Golay derivative window\n")
print(f"{'trailing':>10s} " + "".join(f"{s:>8.1f}s" for s in SGS))
for w in (1, 3, 5, 7, 9):
    row = []
    for sg in SGS:
        s = solve(w, sg)
        row.append("deg" if s is None else str(fields(s)[0]))
    print(f"{('NONE' if w <= 1 else f'{w * 0.1:.1f}s'):>10s} " + "".join(f"{c:>9s}" for c in row))

print("\nrow NONE = no trailing average at all. The control that was missing.\n")
print("TRANSITION ERRORS, to see what the smoothing is actually doing:")
print(f"{'trail':>8s}{'SG':>6s} | {'T1':>7s}{'T2':>7s}{'T3':>7s}{'T4':>7s} | fields")
for w, sg in ((1, 0.3), (1, 0.9), (3, 0.3), (5, 0.3), (5, 0.9), (7, 0.3)):
    s = solve(w, sg)
    if s is None:
        print(f"{w:8d}{sg:6.1f} |  degenerate")
        continue
    k, _ = fields(s)
    errs = [x - tr for x, tr in zip(s[:4], TRT)]
    print(f"{('NONE' if w <= 1 else f'{w * 0.1:.1f}s'):>8s}{sg:6.1f} | "
          + "".join(f"{e:+7.2f}" for e in errs) + f" | {k}/6")
