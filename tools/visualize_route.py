#!/usr/bin/env python
"""
Visualize GPS points and matched route segments for one trajectory.
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple


def parse_linestring(linestring: str) -> List[Tuple[float, float]]:
    """Parse WKT LINESTRING into a list of (lon, lat) tuples."""
    # Expected format: "LINESTRING (lon lat, lon lat, ...)"
    if not isinstance(linestring, str) or "LINESTRING" not in linestring:
        return []
    start = linestring.find("(")
    end = linestring.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return []
    body = linestring[start + 1 : end].strip()
    if not body:
        return []
    points: List[Tuple[float, float]] = []
    for part in body.split(","):
        coords = part.strip().split()
        if len(coords) != 2:
            continue
        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except ValueError:
            continue
        points.append((lon, lat))
    return points


def build_geometry_map(geo_df) -> Dict[int, List[Tuple[float, float]]]:
    """Build fid -> polyline map from edge_geometry.csv dataframe."""
    geo_map: Dict[int, List[Tuple[float, float]]] = {}
    for _, row in geo_df.iterrows():
        try:
            fid = int(row["fid"])
        except (ValueError, TypeError, KeyError):
            continue
        coords = parse_linestring(row.get("geometry", ""))
        if coords:
            geo_map[fid] = coords
    return geo_map


def pick_row(df, tid: str, index: int, seed: int):
    """Pick a row by tid, index, or random seed."""
    if tid:
        match = df[df["tid"] == tid]
        if match.empty:
            raise ValueError(f"tid not found: {tid}")
        return match.iloc[0]
    if index is not None:
        if index < 0 or index >= len(df):
            raise ValueError(f"index out of range: {index}")
        return df.iloc[index]
    rnd = random.Random(seed)
    return df.iloc[rnd.randrange(len(df))]


def flatten_route_segments(route: Sequence[int], geo_map: Dict[int, List[Tuple[float, float]]]):
    """Get list of route polylines in order, skipping missing fids."""
    polylines: List[List[Tuple[float, float]]] = []
    for fid in route:
        if fid in geo_map:
            polylines.append(geo_map[fid])
    return polylines


def iter_segments(polylines: Sequence[Sequence[Tuple[float, float]]]):
    """Yield consecutive segments from polylines."""
    for coords in polylines:
        for i in range(len(coords) - 1):
            yield coords[i], coords[i + 1]


def point_to_segment_distance(p, a, b):
    """Return point-to-segment distance in degrees (lon/lat plane)."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0.0:
        dx, dy = px - ax, py - ay
        return (dx * dx + dy * dy) ** 0.5
    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    projx, projy = ax + t * vx, ay + t * vy
    dx, dy = px - projx, py - projy
    return (dx * dx + dy * dy) ** 0.5


def deg_to_meters(dlon, dlat, lat0):
    """Approximate degree offsets to meters using equirectangular scale."""
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * max(0.0, abs(__import__("math").cos(lat0)))
    return (dlon * meters_per_deg_lon) ** 2 + (dlat * meters_per_deg_lat) ** 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize GPS points and matched route segments.")
    parser.add_argument(
        "--data-path",
        default=os.path.join("data", "chengdu_1101_1115_data_sample10w.pkl"),
        help="Path to data_sample*.pkl",
    )
    parser.add_argument(
        "--edge-geom-path",
        default=os.path.join("data", "edge_geometry.csv"),
        help="Path to edge_geometry.csv",
    )
    parser.add_argument("--tid", default="", help="Trajectory ID to visualize")
    parser.add_argument("--index", type=int, default=None, help="Row index to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when tid/index not provided")
    parser.add_argument(
        "--route",
        choices=["cpath", "opath"],
        default="cpath",
        help="Which route list to draw",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay both cpath and opath for comparison",
    )
    parser.add_argument(
        "--segment-colors",
        action="store_true",
        help="Color adjacent route segments differently",
    )
    parser.add_argument(
        "--no-segment-colors",
        action="store_true",
        help="Disable per-segment coloring (single color per route)",
    )
    parser.add_argument(
        "--segment-boundaries",
        action="store_true",
        help="Mark the start of each route segment",
    )
    parser.add_argument(
        "--boundary-color",
        default="black",
        help="Color for segment boundary markers",
    )
    parser.add_argument(
        "--mode",
        choices=["overlay", "split"],
        default="overlay",
        help="Overlay GPS/route or split into two panels",
    )
    parser.add_argument(
        "--gps-line",
        action="store_true",
        help="Connect GPS points in order to show the trajectory line",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print GPS-to-route distance stats (approx meters)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("tools", "route_visual.png"),
        help="Output image path",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    args = parser.parse_args()
    if not args.no_segment_colors:
        parser.set_defaults(segment_colors=True)

    # Lazy imports so `--help` works without heavy dependencies.
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("Missing dependency. Install from requirements.txt.")
        print(str(exc))
        return 1

    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return 1
    if not os.path.exists(args.edge_geom_path):
        print(f"Edge geometry not found: {args.edge_geom_path}")
        return 1

    df = pd.read_pickle(args.data_path)
    row = pick_row(df, args.tid, args.index, args.seed)

    geo_df = pd.read_csv(args.edge_geom_path)
    geo_map = build_geometry_map(geo_df)

    route_key = "cpath_list" if args.route == "cpath" else "opath_list"
    route = row.get(route_key, [])

    lat_list = row.get("lat_list", [])
    lng_list = row.get("lng_list", [])

    # Filter finite GPS points.
    gps_points = []
    for lat, lng in zip(lat_list, lng_list):
        if lat is None or lng is None:
            continue
        if isinstance(lat, float) and np.isnan(lat):
            continue
        if isinstance(lng, float) and np.isnan(lng):
            continue
        gps_points.append((float(lng), float(lat)))

    polylines = flatten_route_segments(route, geo_map)

    compare_polylines = None
    if args.compare:
        other_key = "opath_list" if route_key == "cpath_list" else "cpath_list"
        other_route = row.get(other_key, [])
        compare_polylines = flatten_route_segments(other_route, geo_map)

    if not gps_points and not polylines:
        print("No GPS points or route geometry found for this row.")
        return 1

    if args.mode == "split":
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
        axes = (ax_left, ax_right)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        axes = (ax,)

    def draw_route(ax, route_polylines, color, label, use_segment_colors, cmap_name, mark_boundaries):
        if not route_polylines:
            return
        if not use_segment_colors:
            for coords in route_polylines:
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.8, label=label)
                if mark_boundaries and coords:
                    ax.scatter([coords[0][0]], [coords[0][1]], s=20, color=args.boundary_color, alpha=0.9)
                label = None
            return
        cmap = plt.get_cmap(cmap_name)
        n = max(1, len(route_polylines))
        for idx, coords in enumerate(route_polylines):
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            seg_color = cmap(idx / max(1, n - 1))
            ax.plot(xs, ys, color=seg_color, linewidth=1.8, alpha=0.9, label=label)
            if mark_boundaries and coords:
                ax.scatter([coords[0][0]], [coords[0][1]], s=20, color=args.boundary_color, alpha=0.9)
            label = None

    use_segment_colors = args.segment_colors and not args.no_segment_colors

    if args.mode == "split":
        # Left: GPS trajectory; Right: route geometry.
        ax_gps, ax_route = axes
        if gps_points:
            xs = [p[0] for p in gps_points]
            ys = [p[1] for p in gps_points]
            ax_gps.scatter(xs, ys, s=8, color="tab:red", alpha=0.8, label="GPS")
            if args.gps_line:
                ax_gps.plot(xs, ys, color="tab:red", linewidth=1.0, alpha=0.6)
            ax_gps.scatter([xs[0]], [ys[0]], s=30, color="tab:green", marker="o", label="Start")
            ax_gps.scatter([xs[-1]], [ys[-1]], s=30, color="tab:purple", marker="x", label="End")
        draw_route(ax_route, polylines, "tab:blue", args.route, use_segment_colors, "viridis", args.segment_boundaries)
        if compare_polylines is not None:
            draw_route(ax_route, compare_polylines, "tab:orange", "compare", use_segment_colors, "plasma", args.segment_boundaries)
        ax_gps.set_title("GPS Trajectory")
        ax_route.set_title("Route Geometry")
        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.legend(loc="best")
    else:
        ax = axes[0]
        # Plot route polylines first for context.
        draw_route(ax, polylines, "tab:blue", args.route, use_segment_colors, "viridis", args.segment_boundaries)
        if compare_polylines is not None:
            draw_route(ax, compare_polylines, "tab:orange", "compare", use_segment_colors, "plasma", args.segment_boundaries)

        # Plot GPS points on top.
        if gps_points:
            xs = [p[0] for p in gps_points]
            ys = [p[1] for p in gps_points]
            ax.scatter(xs, ys, s=8, color="tab:red", alpha=0.8, label="GPS")
            if args.gps_line:
                ax.plot(xs, ys, color="tab:red", linewidth=1.0, alpha=0.6)
            ax.scatter([xs[0]], [ys[0]], s=30, color="tab:green", marker="o", label="Start")
            ax.scatter([xs[-1]], [ys[-1]], s=30, color="tab:purple", marker="x", label="End")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Route vs GPS ({args.route})")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc="best")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)

    print("Saved:", args.out)
    print("tid:", row.get("tid"))
    print("route_key:", route_key, "route_len:", len(route))
    print("gps_len:", len(gps_points))

    if args.metrics and gps_points and polylines:
        segs = list(iter_segments(polylines))
        if segs:
            import math

            lat0 = sum(p[1] for p in gps_points) / len(gps_points)
            distances = []
            for p in gps_points:
                best = None
                for a, b in segs:
                    d = point_to_segment_distance(p, a, b)
                    if best is None or d < best:
                        best = d
                if best is not None:
                    # Convert degree distance to meters (approx).
                    dlon = best
                    dlat = best
                    meters = math.sqrt(deg_to_meters(dlon, dlat, math.radians(lat0)))
                    distances.append(meters)
            if distances:
                distances.sort()
                mean_d = sum(distances) / len(distances)
                p50 = distances[len(distances) // 2]
                p95 = distances[int(len(distances) * 0.95) - 1]
                print("gps_to_route_m: mean=%.2f p50=%.2f p95=%.2f" % (mean_d, p50, p95))

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
