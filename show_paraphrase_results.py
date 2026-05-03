import json
import os
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else 'local_20260502_170115'
base = 'evaluation/paraphrasing_attack'

configs = [
    ('Naive',  'naive/L8'),
    ('G1 U7',  'hi_dypa/G1_U7'),
    ('G2 U6',  'hi_dypa/G2_U6'),
    ('G3 U5',  'hi_dypa/G3_U5'),
    ('G4 U4',  'hi_dypa/G4_U4'),
    ('G5 U3',  'hi_dypa/G5_U3'),
    ('G6 U2',  'hi_dypa/G6_U2'),
    ('G7 U1',  'hi_dypa/G7_U1'),
    ('G8 U0',  'hi_dypa/G8_U0'),
]

print(f"{'Config':<10} {'Acc':>8} {'FPR':>8} {'FNR':>8} {'AvgZ':>8} {'n':>6}")
print('-' * 52)
for label, path in configs:
    p = os.path.join(base, path, tag, 'summary.json')
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        m = s['metrics']
        n = s['total_attack_results']
        print(f"{label:<10} {m['full_identity_accuracy']:>8.4f} {m['false_positive_rate']:>8.4f} {m['false_negative_rate']:>8.4f} {m['avg_z_score']:>8.4f} {n:>6}")
    else:
        print(f"{label:<10} missing")
