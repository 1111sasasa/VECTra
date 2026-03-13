#!/usr/bin/env python
"""Demo runner for gps_segment_audit.py on a small sample."""

import sys

from gps_segment_audit import main


if __name__ == "__main__":
    sys.argv = [
        "gps_segment_audit.py",
        "--sample-rows",
        "2000",
        "--out-dir",
        "tools/audit_out_demo",
    ]
    sys.exit(main())
