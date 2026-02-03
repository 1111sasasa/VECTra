import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


def _build_free_flow_dict(feature_df):
    feature_df = feature_df.copy()
    if (feature_df['road_speed'] <= 0).any():
        feature_df.loc[feature_df['road_speed'] <= 0, 'road_speed'] = feature_df['road_speed'].mean()
    free_flow = (feature_df['length'] / feature_df['road_speed']).astype(float)
    return dict(zip(feature_df['fid'].values.tolist(), free_flow.values.tolist()))


def _build_segment_labels(task_data, max_len=256):
    split_df = task_data.copy()
    split_df['path_len'] = split_df['cpath_list'].map(len)
    split_df = split_df.loc[(split_df['path_len'] > 10) & (split_df['path_len'] < max_len)]

    num_samples = len(split_df)
    y_seg = np.zeros([num_samples, max_len], dtype=np.float32)
    mask = np.zeros([num_samples, max_len], dtype=np.float32)
    route_ids = np.full([num_samples, max_len], -1, dtype=np.int32)

    for i in range(num_samples):
        row = split_df.iloc[i]
        path = np.array(row['cpath_list'], dtype=np.int32)
        path_len = len(path)
        if 'road_interval' in row and row['road_interval'] is not None:
            interval = np.array(row['road_interval'], dtype=np.float32)
        else:
            ts = np.array(row['road_timestamp'], dtype=np.float32)
            interval = np.diff(ts)
        if len(interval) != path_len:
            path_len = min(path_len, len(interval))
            path = path[:path_len]
            interval = interval[:path_len]
        path_len = min(path_len, max_len)
        y_seg[i, :path_len] = interval[:path_len]
        mask[i, :path_len] = 1.0
        route_ids[i, :path_len] = path[:path_len]

    return split_df, y_seg, mask, route_ids


def _build_segment_labels_from_df(split_df, max_len=256):
    split_df = split_df.copy()
    if 'path_len' not in split_df:
        split_df['path_len'] = split_df['cpath_list'].map(len)

    num_samples = len(split_df)
    y_seg = np.zeros([num_samples, max_len], dtype=np.float32)
    mask = np.zeros([num_samples, max_len], dtype=np.float32)
    route_ids = np.full([num_samples, max_len], -1, dtype=np.int32)

    for i in range(num_samples):
        row = split_df.iloc[i]
        path = np.array(row['cpath_list'], dtype=np.int32)
        path_len = len(path)
        if 'road_interval' in row and row['road_interval'] is not None:
            interval = np.array(row['road_interval'], dtype=np.float32)
        else:
            ts = np.array(row['road_timestamp'], dtype=np.float32)
            interval = np.diff(ts)
        if len(interval) != path_len:
            path_len = min(path_len, len(interval))
            path = path[:path_len]
            interval = interval[:path_len]
        path_len = min(path_len, max_len)
        y_seg[i, :path_len] = interval[:path_len]
        mask[i, :path_len] = 1.0
        route_ids[i, :path_len] = path[:path_len]

    return split_df, y_seg, mask, route_ids


def _bucketize(index, bins):
    # bins: list of thresholds for congestion index
    labels = np.zeros_like(index, dtype=np.int64)
    for i, b in enumerate(bins):
        labels[index >= b] = i + 1
    return labels


def evaluation(segment_time_pred, task_data, feature_df, bins=(1.2, 1.5, 2.0), split_df=None):
    # segment_time_pred: (num_samples, max_len) torch or numpy
    if split_df is None:
        split_df, y_true, mask, route_ids = _build_segment_labels(
            task_data, max_len=segment_time_pred.shape[1]
        )
    else:
        split_df, y_true, mask, route_ids = _build_segment_labels_from_df(
            split_df, max_len=segment_time_pred.shape[1]
        )
    free_flow_dict = _build_free_flow_dict(feature_df)
    default_free_flow = float(np.mean(list(free_flow_dict.values())))

    if torch.is_tensor(segment_time_pred):
        y_pred = segment_time_pred.detach().cpu().numpy()
    else:
        y_pred = np.asarray(segment_time_pred)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            "segment_time_pred count does not match label rows. "
            "Pass seg_result['split_df'] to align filtered samples."
        )

    free_flow = np.full_like(y_true, default_free_flow, dtype=np.float32)
    for i in range(route_ids.shape[0]):
        for j in range(route_ids.shape[1]):
            rid = route_ids[i, j]
            if rid < 0:
                continue
            free_flow[i, j] = free_flow_dict.get(rid, default_free_flow)

    true_index = y_true / np.clip(free_flow, 1e-6, None)
    pred_index = y_pred / np.clip(free_flow, 1e-6, None)

    valid = mask.astype(bool)
    true_idx_valid = true_index[valid]
    pred_idx_valid = pred_index[valid]

    mae = mean_absolute_error(true_idx_valid, pred_idx_valid)
    rmse = mean_squared_error(true_idx_valid, pred_idx_valid) ** 0.5

    true_label = _bucketize(true_idx_valid, bins)
    pred_label = _bucketize(pred_idx_valid, bins)
    macro_f1 = f1_score(true_label, pred_label, average='macro')
    micro_f1 = f1_score(true_label, pred_label, average='micro')

    print(
        f'congestion inference | index MAE: {mae:.4f}, RMSE: {rmse:.4f}, '
        f'micro F1: {micro_f1:.4f}, macro F1: {macro_f1:.4f}'
    )

    return {
        'index_mae': mae,
        'index_rmse': rmse,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'bins': bins
    }
