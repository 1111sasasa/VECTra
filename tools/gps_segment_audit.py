#!/usr/bin/env python
"""Audit GPS-to-segment mapping and segment reuse across trajectories."""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit GPS-to-segment mapping and segment reuse.")
    parser.add_argument(
        "--data-path",
        default=os.path.join("data", "chengdu_1101_1115_data_sample10w.pkl"),
        help="Path to data_sample*.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("tools", "audit_out"),
        help="Output directory for CSV reports",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="If >0, only analyze the first N rows",
    )
    parser.add_argument(
        "--gps-round",
        type=int,
        default=5,
        help="Decimal places to round GPS points when comparing across trajectories",
    )
    parser.add_argument(
        "--sample-per-seg",
        type=int,
        default=5,
        help="Max sample tids/GPS points to keep per segment",
    )
    parser.add_argument(
        "--tid",
        default="",
        help="If provided, export GPS-to-segment mapping for this trajectory",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        print("Missing dependency. Install from requirements.txt.")
        print(str(exc))
        return 1

    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return 1

    df = pd.read_pickle(args.data_path)
    if args.sample_rows > 0:
        df = df.iloc[: args.sample_rows].reset_index(drop=True)

    required_cols = {"tid", "opath_list", "lat_list", "lng_list"}
    missing = required_cols - set(df.columns)
    if missing:
        print("Missing columns:", sorted(missing))
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    # Segment reuse stats.
    seg_tids = defaultdict(set)
    seg_gps_count = defaultdict(int)
    seg_tid_samples = defaultdict(list)
    seg_gps_samples = defaultdict(list)

    # GPS-point reuse across trajectories.
    gps_to_seg = defaultdict(set)

    # Optional per-trajectory mapping.
    tid_mapping_rows = []

    for _, row in df.iterrows():
        tid = row["tid"]
        opath = row["opath_list"]
        lat_list = row["lat_list"]
        lng_list = row["lng_list"]

        if not opath or not lat_list or not lng_list:
            continue
        n = min(len(opath), len(lat_list), len(lng_list))
        if n == 0:
            continue

        for i in range(n):
            seg_id = _normalize_seg_id(opath[i])
            lat = lat_list[i]
            lng = lng_list[i]
            if lat is None or lng is None or _is_nan(lat) or _is_nan(lng):
                continue
            seg_tids[seg_id].add(tid)
            seg_gps_count[seg_id] += 1

            if len(seg_tid_samples[seg_id]) < args.sample_per_seg:
                seg_tid_samples[seg_id].append(tid)
            if len(seg_gps_samples[seg_id]) < args.sample_per_seg:
                seg_gps_samples[seg_id].append((float(lat), float(lng)))

            key = (round(float(lat), args.gps_round), round(float(lng), args.gps_round))
            gps_to_seg[key].add(seg_id)

            if args.tid and tid == args.tid:
                tid_mapping_rows.append(
                    {
                        "idx": i,
                        "lat": float(lat),
                        "lng": float(lng),
                        "seg_id": seg_id,
                    }
                )

    # Build segment reuse report.
    seg_rows = []
    for seg_id, tids in seg_tids.items():
        seg_rows.append(
            {
                "seg_id": seg_id,
                "tid_count": len(tids),
                "gps_point_count": seg_gps_count.get(seg_id, 0),
                "sample_tids": ";".join(str(t) for t in seg_tid_samples.get(seg_id, [])),
                "sample_gps": ";".join(
                    f"{lat},{lng}" for lat, lng in seg_gps_samples.get(seg_id, [])
                ),
            }
        )
    seg_df = pd.DataFrame(seg_rows).sort_values(["tid_count", "gps_point_count"], ascending=False)
    seg_out = os.path.join(args.out_dir, "segment_reuse.csv")
    seg_out = _safe_to_csv(seg_df, seg_out)

    # GPS points mapped to multiple segments.
    multi_rows = []
    for (lat, lng), segs in gps_to_seg.items():
        if len(segs) > 1:
            multi_rows.append(
                {
                    "lat": lat,
                    "lng": lng,
                    "segment_ids": ";".join(str(s) for s in sorted(segs)),
                    "segment_count": len(segs),
                }
            )
    multi_df = pd.DataFrame(multi_rows).sort_values("segment_count", ascending=False)
    multi_out = os.path.join(args.out_dir, "gps_point_multi_segment.csv")
    multi_out = _safe_to_csv(multi_df, multi_out)

    if args.tid:
        tid_df = pd.DataFrame(tid_mapping_rows)
        tid_out = os.path.join(args.out_dir, f"gps_to_segment_{args.tid}.csv")
        tid_out = _safe_to_csv(tid_df, tid_out)
    else:
        tid_out = ""

    # Summary.
    print("rows:", len(df))
    print("segments:", len(seg_tids))
    print("segment_reuse.csv:", seg_out)
    print("gps_point_multi_segment.csv:", multi_out)
    if tid_out:
        print("gps_to_segment.csv:", tid_out)
    if len(multi_df) > 0:
        print("multi_segment_gps_points:", len(multi_df))
    else:
        print("multi_segment_gps_points: 0")

    return 0


def _normalize_seg_id(seg_id):
    try:
        return int(seg_id)
    except (ValueError, TypeError):
        return str(seg_id)


def _is_nan(value):
    try:
        return value != value
    except Exception:
        return False


def _safe_to_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"{base}_{ts}{ext}"
        df.to_csv(alt, index=False)
        return alt


if __name__ == "__main__":
    sys.exit(main())
