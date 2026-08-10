#!/usr/bin/env python3
# Simple, locked results analysis (2026-08-10). TWO truth-anchored metrics only:
#   (1) "was it genuinely risky?" = clean-spine Pc at the DECISION point (root),
#       from the wait_spine (zero-innovation / perfect-tracking) curve — NOT the
#       tautological belief pc (which is ~0 by construction since MCTS burns
#       whenever its belief would exceed δ).
#   (2) "did the choice pan out?" = the TRUE relative miss distance (miss_m), the
#       actual geometry, no belief involved.
# Per-policy result = the simple story: waits when it can (defer + stayed safe),
# burns when it must, fixed rules can't tell the difference.
#
# Data: MCTS traces (figureScripts/data/traces/spacecraftCA-mcts-clean/*.json) for
# spine + true miss; sweep_all.csv for the per-policy scalars (all policies).
import json, glob, csv
from collections import defaultdict

THR = 1e-5
TRACES = 'figureScripts/data/traces/spacecraftCA-mcts-clean'

def spine_root_pc(d):
    # clean-spine Pc at the DECISION point = the max over the spine (earliest/widest),
    # i.e. "how risky before any measurement resolves it". Use the first (root) epoch.
    c=d['wait_spine']['columns']; rows=d['wait_spine']['rows']
    vals=[dict(zip(c,r))['wait_spine_pc'] for r in rows]
    return vals[0], max(vals)   # (root, max) — root is at detection

def load_mcts_traces():
    out={}
    for f in glob.glob(f'{TRACES}/*.json'):
        d=json.load(open(f))
        tc=d['trace']['columns']; tr=d['trace']['rows']
        last=dict(zip(tc,tr[-1]))
        n_man=sum(1 for r in tr if dict(zip(tc,r))['action']=='MANEUVER')
        root_pc,_=spine_root_pc(d)
        key=(d['case_id'], d['sensor_quality'], str(d['cadence_h']), str(d['seed']))
        out[key]=dict(spine_root_pc=root_pc, true_miss=last['miss_m'],
                      action=('MANEUVER' if n_man>0 else 'DEFER'), n_man=n_man)
    return out

mcts=load_mcts_traces()
print(f"MCTS traces: {len(mcts)}")

# --- METRIC 1: how many conjunctions are GENUINELY RISKY (spine root Pc > THR)? ---
risky=[k for k,v in mcts.items() if v['spine_root_pc']>THR]
print(f"\n[Metric 1] genuinely-risky (clean-spine root Pc > {THR:g}): {len(risky)}/{len(mcts)}")

# --- METRIC 2: on MCTS DEFER episodes, did the true miss stay safe? ---
defers=[v for v in mcts.values() if v['action']=='DEFER']
mans  =[v for v in mcts.values() if v['action']=='MANEUVER']
print(f"\n[MCTS] deferred {len(defers)}/{len(mcts)} ({len(defers)/len(mcts)*100:.0f}%), maneuvered {len(mans)}")
if defers:
    dm=sorted(v['true_miss'] for v in defers)
    print(f"  DEFER true miss: min={dm[0]:.0f} m  median={dm[len(dm)//2]:.0f} m  (all deferrals — did waiting stay safe?)")
    # a deferral 'stayed safe' if true miss comfortably outside HBR — report min as the worst case
# risky AND deferred = the interesting cases (MCTS judged a real threat resolvable by waiting)
risky_defer=[(k,mcts[k]) for k in risky if mcts[k]['action']=='DEFER']
print(f"  of the {len(risky)} risky cases, MCTS deferred {len(risky_defer)}; their true miss:")
for k,v in sorted(risky_defer, key=lambda x:x[1]['true_miss'])[:10]:
    print(f"    {k[0][-5:]} {k[1]:6} {k[2]:>3}h s{k[3]}  spine_root_pc={v['spine_root_pc']:.2e}  true_miss={v['true_miss']:.0f} m")

# --- per-policy TRUE-MISS from the scalar csv (all policies, the headline contrast) ---
print("\n[All policies] true-miss safety from sweep_all.csv (the fixed-rule contrast):")
rows=[r for r in csv.DictReader(open('figureScripts/data/sweep_all.csv')) if r['state']=='finished']
def num(x):
    try: return float(x)
    except: return None
print(f"  {'policy':16} {'defer%':>7} {'min_true_miss':>14} {'#<100m(true)':>13} {'meanDv':>8}")
for p in ['delay_28h','delay_12h','delay_6h','delay_3h','mcts','wait_feasibility']:
    sub=[r for r in rows if r['policy_variant']==p]
    if not sub: continue
    n=len(sub)
    defer=sum(1 for r in sub if r['resolved_without_maneuver'].lower() in('true','1'))/n*100
    miss=[num(r['final_miss_m']) for r in sub if num(r['final_miss_m']) is not None]
    dv=[num(r['total_dv_mps']) for r in sub if num(r['total_dv_mps']) is not None]
    print(f"  {p:16} {defer:6.1f}% {min(miss):13.0f}m {sum(1 for m in miss if m<100):13d} {sum(dv)/len(dv):8.3f}")

# --- SANITY / anomaly flags ---
print("\n[sanity] anomaly checks:")
neg=[k for k,v in mcts.items() if v['true_miss']<0]
print(f"  negative true miss: {len(neg)} (should be 0)")
defer_close=[(k,v) for k,v in mcts.items() if v['action']=='DEFER' and v['true_miss']<50]
print(f"  MCTS DEFER with true miss <50 m (would be a dangerous deferral): {len(defer_close)}")
for k,v in defer_close: print(f"    !! {k}  true_miss={v['true_miss']:.1f} m spine_root_pc={v['spine_root_pc']:.2e}")

# =========================================================================
# MANEUVER-SIDE breakdown: was each MCTS burn NECESSARY? (added 2026-08-10)
# A burn is NECESSARY if the clean spine never goes durably-safe (waiting could
# not have resolved it). A POSSIBLE OVER-MANEUVER is a burn where the clean spine
# WAS durably-safe (perfect tracking would have deferred) — expected to be
# noise-driven, concentrated at poor tracking.
# =========================================================================
def _spine_all(d):
    c=d['wait_spine']['columns']; rows=d['wait_spine']['rows']
    return [dict(zip(c,r))['wait_spine_pc'] for r in rows]

_mans=[]; _defs=[]
for f in glob.glob(f'{TRACES}/*.json'):
    d=json.load(open(f)); tc=d['trace']['columns']; tr=d['trace']['rows']
    n_man=sum(1 for r in tr if dict(zip(tc,r))['action']=='MANEUVER')
    sv=_spine_all(d)
    rec=dict(q=d['sensor_quality'], durably_safe=(max(sv)<=THR), root_pc=sv[0])
    (_mans if n_man>0 else _defs).append(rec)

_nec=[m for m in _mans if not m['durably_safe']]
_over=[m for m in _mans if m['durably_safe']]
print("\n[Maneuver side] was each MCTS burn necessary?")
print(f"  necessary (spine never durably-safe): {len(_nec)}/{len(_mans)} ({len(_nec)/len(_mans)*100:.0f}%)")
print(f"  possible over-maneuver (spine WAS durably-safe): {len(_over)}/{len(_mans)} ({len(_over)/len(_mans)*100:.0f}%)")
_byq=defaultdict(int)
for m in _over: _byq[m['q']]+=1
print("  over-maneuvers by tracking quality (noise-driven → rises as tracking degrades):")
for q in ['best','median','worst']: print(f"    {q}: {_byq[q]}")
