K1 dribble v13b: arms kept down at low ball speed. From policies/k1_dribble_v11_handover/model_33500.pt,
fine-tuned 600 iterations with MJLAB_K1_DRIBBLE_ARMS_LOW=-0.1,0.5 (shoulders and elbows past 0.5 rad
from the arms-down default, penalised in full below 0.5 m/s, fading out by 1.0 m/s), entropy 0, the
same settings as v11 otherwise (CURVES, TIGHT, SPRINT, receive/first-touch hand-over bank).
Source logs/rsl_rl/k1_dribble/2026-11-20_22-20-33/. Script scripts/dribbling/run_dribble_arms_low.sh.

Use model_33700.pt (2026-11-20). Against v11 33500 (same test runs):
  elbow bend at 0.3 m/s, mean rad from default: pitch 0.65 -> 0.12, yaw 1.18 -> 0.64
  shoulders: mean 0.19/0.10 -> 0.21/0.21 rad (pitch/roll), still inside the free 0.5 rad
  ball speed at 0.6/1.0/1.3/1.8 m/s: 0.69/1.08/1.30/1.54 (v11 0.72/1.10/1.28/1.51), no falls
  figure of eight 0.8/0.6/0.5/0.4 m: 100/92/83/66% (v11 in the same run 95/82/71/53%)
  walk -> dribble: 0.9-1.0% fell, 98-99% kept (v11 1.2-1.8%)
  dribble -> walk 0.2% fell; receive -> dribble 0% fell, 100% kept; first touch v5 -> dribble 1% fell, 97% kept
  kick v10 from v13b dribble states (state_banks/k1_dribble_bank_v13b.pt): 99% struck, 1% fell
The later checkpoints (33800, 33900, 34000) failed walk -> dribble (17-62% fell), 34099 recovered;
the first try (weight -0.5, deadband 0.3, entropy 0.002) collapsed.
