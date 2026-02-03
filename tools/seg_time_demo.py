import numpy as np
import pandas as pd
import torch

# Add project root to sys.path for direct script execution.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task import seg_time, congestion_inf


def build_synthetic_data(num_samples=50, max_len=64, num_roads=100, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_samples):
        path_len = int(rng.integers(12, min(30, max_len)))
        cpath_list = rng.integers(0, num_roads, size=path_len).tolist()
        road_interval = rng.uniform(5.0, 60.0, size=path_len).tolist()
        start_time = float(rng.integers(1_700_000_000, 1_700_100_000))
        ts = [start_time]
        for t in road_interval:
            ts.append(ts[-1] + float(t))
        total_time = float(sum(road_interval))
        rows.append({
            'cpath_list': cpath_list,
            'road_interval': road_interval,
            'road_timestamp': ts,
            'total_time': total_time,
            'start_time': start_time
        })
    return pd.DataFrame(rows)


def build_feature_df(num_roads=100, seed=0):
    rng = np.random.default_rng(seed)
    length = rng.uniform(30.0, 500.0, size=num_roads)
    road_speed = rng.uniform(5.0, 20.0, size=num_roads)
    return pd.DataFrame({
        'fid': np.arange(num_roads, dtype=int),
        'length': length,
        'road_speed': road_speed
    })


def main():
    num_samples = 50
    max_len = 64
    num_roads = 100
    emb_dim = 16

    task_data = build_synthetic_data(num_samples=num_samples, max_len=max_len, num_roads=num_roads)
    feature_df = build_feature_df(num_roads=num_roads)

    route_road_rep = torch.randn(num_samples, max_len, emb_dim)

    result = seg_time.evaluation(route_road_rep, task_data, num_nodes=num_roads, fold=2, epoch_num=3)
    congestion_inf.evaluation(result['segment_time_pred'], task_data, feature_df)


if __name__ == '__main__':
    main()
