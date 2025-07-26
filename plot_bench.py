import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Paths for the newly uploaded "low‑rated" CSVs
path_a = Path("/mnt/data/results_5010000vecs_low_rated.csv")
path_b = Path("/mnt/data/results_5010000vecs_low_rated (1).csv")

# Load and detect which is CAGRA vs HNSW
df_a = pd.read_csv(path_a)
df_b = pd.read_csv(path_b)

def detect_algo(df):
    if 'intermediate_graph_degree' in df.columns:
        return 'CAGRA'
    elif 'M' in df.columns and 'efConstruction' in df.columns:
        return 'HNSW'
    else:
        return 'UNKNOWN'

algo_a = detect_algo(df_a)
algo_b = detect_algo(df_b)

# Map column names
def harmonise(df, algo_name):
    if 'latency_ms' not in df.columns and 'avg_query_latency_ms' in df.columns:
        df = df.rename(columns={'avg_query_latency_ms': 'latency_ms'})
    if 'queries_per_second' in df.columns:
        df = df.rename(columns={'queries_per_second': 'qps'})
    df = df.rename(columns={'latency_ms':'latency_ms'})
    df['algo'] = algo_name
    needed = ['recall', 'build_time_seconds', 'latency_ms', 'qps', 'algo']
    return df[needed]

df_a_h = harmonise(df_a, algo_a)
df_b_h = harmonise(df_b, algo_b)
df = pd.concat([df_a_h, df_b_h], ignore_index=True)

# ---------- 1. Build‑Time bar vs recall bucket ----------
bins   = [0.80, 0.90, 0.95, 0.99, 1.01]
labels = ['80–90%', '90–95%', '95–99%', '≥99%']
df['recall_bin'] = pd.cut(df['recall'], bins=bins, labels=labels,
                          right=False, include_lowest=True)
summary = (df.dropna(subset=['recall_bin'])
             .groupby(['recall_bin','algo'])['build_time_seconds']
             .mean()
             .unstack())

x = np.arange(len(labels))
width=0.35
plt.figure(figsize=(8,5))
plt.bar(x-width/2, summary.get('CAGRA', pd.Series([np.nan]*len(labels))), width, label='CAGRA')
plt.bar(x+width/2, summary.get('HNSW', pd.Series([np.nan]*len(labels))),  width, label='HNSW')
plt.ylabel('Average Build Time (s)')
plt.xlabel('Recall Range')
plt.xticks(x, labels)
plt.title('Build Time vs Recall Range (Low‑rated subset)')
plt.grid(axis='y', linestyle='--', alpha=.4)
plt.legend()
plt.tight_layout()
build_out = Path('/mnt/data/build_time_vs_recall_low.png')
plt.savefig(build_out, dpi=120)
plt.close()

# ---------- 2. Latency vs recall ----------
plt.figure(figsize=(7,5))
markers={'CAGRA':'o','HNSW':'^'}
for algo,grp in df.groupby('algo'):
    grp=grp.sort_values('recall')
    plt.plot(grp['recall'], grp['latency_ms'],
             marker=markers.get(algo,'o'), linewidth=2, label=algo)
plt.xlabel('Recall')
plt.ylabel('Average Search Latency (ms)')
plt.title('Latency vs Recall (Low‑rated subset)')
plt.grid(True, linestyle='--', alpha=.4)
plt.legend()
plt.tight_layout()
lat_out = Path('/mnt/data/latency_vs_recall_low.png')
plt.savefig(lat_out, dpi=120)
plt.close()

# ---------- 3. Throughput vs recall ----------
plt.figure(figsize=(7,5))
for algo,grp in df.groupby('algo'):
    grp=grp.sort_values('recall')
    plt.plot(grp['recall'], grp['qps'],
             marker=markers.get(algo,'o'), linewidth=2, label=algo)
plt.xlabel('Recall')
plt.ylabel('Throughput (queries / second)')
plt.title('Throughput vs Recall (Low‑rated subset)')
plt.grid(True, linestyle='--', alpha=.4)
plt.legend()
plt.tight_layout()
thr_out = Path('/mnt/data/throughput_vs_recall_low.png')
plt.savefig(thr_out, dpi=120)
plt.close()

(build_out, lat_out, thr_out)
