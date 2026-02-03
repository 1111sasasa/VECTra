import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time


def next_batch_index(ds, bs, shuffle=True):
    num_batches = math.ceil(ds / bs)
    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)
    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i:]
        else:
            batch_index = index[bs * i: bs * (i + 1)]
        yield batch_index


class SegTimeHead(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch, max_len, dim)
        bsz, max_len, dim = x.shape
        x = x.view(bsz * max_len, dim)
        y = self.net(x)
        return y.view(bsz, max_len)


def _build_segment_labels(task_data, padding_id, min_len=10, max_len=256):
    # task_data needs: cpath_list, road_interval or road_timestamp, total_time, start_time
    split_df = task_data.copy()
    split_df['path_len'] = split_df['cpath_list'].map(len)
    split_df = split_df.loc[(split_df['path_len'] > min_len) & (split_df['path_len'] < max_len)]
    kept_indices = split_df.index.to_numpy()
    split_df = split_df.reset_index(drop=True)

    num_samples = len(split_df)
    y_seg = np.zeros([num_samples, max_len], dtype=np.float32)
    mask = np.zeros([num_samples, max_len], dtype=np.float32)
    total_time = np.zeros([num_samples], dtype=np.float32)
    start_time = np.zeros([num_samples], dtype=np.float32)

    for i in range(num_samples):
        row = split_df.iloc[i]
        path_len = row['path_len']
        if 'road_interval' in row and row['road_interval'] is not None:
            interval = np.array(row['road_interval'], dtype=np.float32)
        else:
            ts = np.array(row['road_timestamp'], dtype=np.float32)
            interval = np.diff(ts)
        if len(interval) != path_len:
            path_len = min(path_len, len(interval))
            interval = interval[:path_len]
        y_seg[i, :path_len] = interval
        mask[i, :path_len] = 1.0
        if 'total_time' in row and not np.isnan(row['total_time']):
            total_time[i] = float(row['total_time'])
        else:
            total_time[i] = float(np.sum(interval))
        if 'start_time' in row and not np.isnan(row['start_time']):
            start_time[i] = float(row['start_time'])
        elif 'road_timestamp' in row and row['road_timestamp'] is not None:
            start_time[i] = float(row['road_timestamp'][0])
        else:
            start_time[i] = 0.0

    return (
        torch.FloatTensor(y_seg),
        torch.FloatTensor(mask),
        torch.FloatTensor(total_time),
        torch.FloatTensor(start_time),
        split_df,
        kept_indices
    )


def _normalize_to_total_time(raw, total_time, eps=1e-6):
    # raw: (batch, max_len)
    positive = torch.nn.functional.softplus(raw)
    denom = positive.sum(dim=1, keepdim=True).clamp_min(eps)
    scale = total_time.unsqueeze(1)
    return positive / denom * scale


def _masked_mse(pred, label, mask):
    diff = (pred - label) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def _masked_mae(pred, label, mask):
    diff = (pred - label).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def compute_arrival_times(start_time, seg_time):
    # start_time: (batch,), seg_time: (batch, max_len)
    start_time = start_time.unsqueeze(1)
    arrival = torch.cumsum(seg_time, dim=1)
    return torch.cat([start_time, start_time + arrival], dim=1)


def evaluation(route_road_rep, task_data, num_nodes, fold=5, epoch_num=50, batch_size=64, eval_batch_size=None,
               device=None, log_interval=5, use_amp=False, align_log=True):
    # route_road_rep: (num_samples, max_len, dim)
    if eval_batch_size is None:
        eval_batch_size = batch_size
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if route_road_rep.is_cuda:
        # Keep large route representations on CPU to reduce peak GPU memory.
        route_road_rep = route_road_rep.detach().cpu()

    max_len = route_road_rep.shape[1]
    y_seg, mask, total_time, start_time, split_df, kept_indices = _build_segment_labels(
        task_data, num_nodes, max_len=max_len
    )

    idx = torch.as_tensor(kept_indices, dtype=torch.long)
    x = route_road_rep.index_select(0, idx)

    split = x.shape[0] // fold
    usable_count = split * fold
    if align_log:
        total_samples = len(task_data)
        kept_count = len(kept_indices)
        sample_idx = kept_indices[:5].tolist()
        path_lens = task_data.iloc[kept_indices[:5]]['cpath_list'].map(len).tolist()
        print(
            "[seg_time][align] total={} kept={} route_rep={} x={} max_len={} fold={} usable={}".format(
                total_samples, kept_count, route_road_rep.shape[0], x.shape[0], max_len, fold, usable_count
            )
        )
        print("[seg_time][align] kept_indices_head={} path_len_head={}".format(sample_idx, path_lens))

    if usable_count < x.shape[0]:
        # Drop tail samples so predictions and labels share the same aligned set.
        x = x[:usable_count]
        y_seg = y_seg[:usable_count]
        mask = mask[:usable_count]
        total_time = total_time[:usable_count]
        start_time = start_time[:usable_count]
        kept_indices = kept_indices[:usable_count]
        split_df = split_df.iloc[:usable_count].reset_index(drop=True)

    fold_preds, fold_trues, fold_masks = [], [], []
    for i in range(fold):
        fold_start = time.time()
        eval_idx = list(range(i * split, (i + 1) * split, 1))
        train_idx = list(set(list(range(x.shape[0]))) - set(eval_idx))

        x_train, x_eval = x[train_idx], x[eval_idx]
        y_train, y_eval = y_seg[train_idx], y_seg[eval_idx]
        m_train, m_eval = mask[train_idx], mask[eval_idx]
        t_train, t_eval = total_time[train_idx], total_time[eval_idx]

        model = SegTimeHead(x.shape[2]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

        print(f"[seg_time] fold {i + 1}/{fold} train={x_train.shape[0]} eval={x_eval.shape[0]}")
        best_mae = 1e9
        best_pred = None
        for epoch in range(1, epoch_num + 1):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            for batch_index in next_batch_index(x_train.shape[0], batch_size):
                opt.zero_grad(set_to_none=True)
                x_batch = x_train[batch_index].to(device, non_blocking=True)
                y_batch = y_train[batch_index].to(device, non_blocking=True)
                m_batch = m_train[batch_index].to(device, non_blocking=True)
                t_batch = t_train[batch_index].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    raw = model(x_batch)
                    pred = _normalize_to_total_time(raw, t_batch)
                    loss = _masked_mse(pred, y_batch, m_batch)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                epoch_loss += loss.detach().item()
                batch_count += 1

            model.eval()
            with torch.no_grad():
                pred_eval_list = []
                for batch_index in next_batch_index(x_eval.shape[0], eval_batch_size, shuffle=False):
                    x_batch = x_eval[batch_index].to(device, non_blocking=True)
                    t_batch = t_eval[batch_index].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        pred = _normalize_to_total_time(model(x_batch), t_batch)
                    pred_eval_list.append(pred.detach().cpu())
                pred_eval = torch.cat(pred_eval_list, dim=0)
                mae = _masked_mae(pred_eval, y_eval, m_eval).item()
                if mae < best_mae:
                    best_mae = mae
                    best_pred = pred_eval.detach().cpu()

            if epoch == 1 or epoch == epoch_num or (log_interval and epoch % log_interval == 0):
                avg_loss = epoch_loss / max(1, batch_count)
                cost = time.time() - epoch_start
                print(f"[seg_time] fold {i + 1}/{fold} epoch {epoch}/{epoch_num} loss={avg_loss:.4f} best_mae={best_mae:.4f} cost={cost:.1f}s")

        print(f"[seg_time] fold {i + 1}/{fold} done, best_mae={best_mae:.4f}, cost={time.time() - fold_start:.1f}s")
        fold_preds.append(best_pred)
        fold_trues.append(y_eval.detach().cpu())
        fold_masks.append(m_eval.detach().cpu())

    y_preds = torch.cat(fold_preds, dim=0)
    y_trues = torch.cat(fold_trues, dim=0)
    y_masks = torch.cat(fold_masks, dim=0)

    mae = _masked_mae(y_preds, y_trues, y_masks).item()
    rmse = torch.sqrt(_masked_mse(y_preds, y_trues, y_masks)).item()
    print(f'segment time decomposition | MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    pred_len = y_preds.shape[0]
    arrival_time = compute_arrival_times(start_time[:pred_len], y_preds)
    split_df_used = task_data.iloc[kept_indices[:pred_len]].reset_index(drop=True)

    return {
        'segment_time_pred': y_preds,
        'segment_time_true': y_trues,
        'segment_mask': y_masks,
        'arrival_time_pred': arrival_time,
        'split_df': split_df_used
    }
