# Visualization Tools

## Route vs GPS

Create a plot of GPS points and matched route segments for one trajectory.

### Install

- Use the root requirements file: `requirements.txt`.

### Run

```powershell
python tools\visualize_route.py --help
```

```powershell
python tools\visualize_route.py --index 0 --out tools\route_visual.png
```

```powershell
python tools\visualize_route.py --index 0 --route cpath --gps-line --metrics --show
```

```powershell
python tools\visualize_route.py --index 0 --compare --mode split --out tools\route_compare.png
```

```powershell
python tools\visualize_route.py --tid <your_tid> --route cpath --show
```

## GPS to Segment Audit

Generate CSV reports to check whether the same GPS points map to one or multiple segment IDs across trajectories.

### Run

```powershell
python tools\gps_segment_audit.py --sample-rows 2000 --out-dir tools\audit_out
```

```powershell
python tools\gps_segment_audit.py --tid <your_tid> --out-dir tools\audit_out
```

```powershell
python tools\gps_segment_audit_demo.py
```

### Outputs

- `tools/audit_out/segment_reuse.csv`: per-segment tid counts and sample GPS points
- `tools/audit_out/gps_point_multi_segment.csv`: GPS points mapped to multiple segments
- `tools/audit_out/gps_to_segment_<tid>.csv`: GPS-to-segment mapping for one trajectory
