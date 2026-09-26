import numpy as np, sweep as S
best=(3,0.3)
s=S.solve(*best); T1,T2,T3,T4,cc,acd,nxt=s
d={"avg_digging":T2-T1,"avg_hauling":T3-T2,"avg_dumping":T4-T3,"avg_swinging":nxt-T4}
TR={"avg_digging":S.TRD["dig"],"avg_hauling":S.TRD["haul"],"avg_dumping":S.TRD["dump"],"avg_swinging":S.TRD["swing"]}
print(f"BEST: trailing {best[0]*0.1:.1f}s + Savitzky-Golay {best[1]:.1f}s\n")
print(f"{'transition':12s} {'pred':>7s} {'true':>7s} {'err':>7s}")
for nm,p,tr in (("T1 digging",T1,S.TRT[0]),("T2 hauling",T2,S.TRT[1]),("T3 dumping",T3,S.TRT[2]),("T4 swinging",T4,S.TRT[3])):
    print(f"{nm:12s} {p:7.2f} {tr:7.2f} {p-tr:+7.2f}")
print(f"\n{'GRADED FIELD':22s} {'pred':>8s} {'true':>8s} {'err':>7s}  margin left")
print(f"{'cycle_count':22s} {cc:8d} {1:8d} {'exact':>7s}  n/a")
e=acd-S.TRD["cyc"]
print(f"{'avg_cycle_duration':22s} {acd:8.2f} {S.TRD['cyc']:8.2f} {e:+7.2f}  {0.6-abs(e):.2f}s")
ok=2
for k in d:
    e=d[k]-TR[k]; ok+=abs(e)<=0.6
    print(f"{k:22s} {d[k]:8.2f} {TR[k]:8.2f} {e:+7.2f}  {0.6-abs(e):+.2f}s")
print(f"\n{ok}/6")
print("\nhow many of the 25 grid cells reach each score:")
import collections
c=collections.Counter()
for w in (1,3,5,7,9):
    for sg in S.SGS:
        r=S.solve(w,sg); c[("deg" if r is None else S.fields(r)[0])]+=1
for k in sorted(c,key=str): print(f"   {k}: {c[k]} cells")
