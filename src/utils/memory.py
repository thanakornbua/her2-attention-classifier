import csv
import os
import time
import resource

try:
    import psutil
except Exception:
    psutil = None


def get_memory_info():
    """Return a small dict with current process memory info.

    Fields:
      - timestamp: unix epoch float
      - rss: resident set size in bytes (or None)
      - vms: virtual memory size in bytes (or None)
      - percent: process memory percent (or None)
      - ru_maxrss: peak RSS (platform dependent units; on Linux it's KB) or None
    """
    ts = time.time()
    rss = vms = percent = None
    try:
        if psutil:
            p = psutil.Process()
            mi = p.memory_info()
            rss = getattr(mi, 'rss', None)
            vms = getattr(mi, 'vms', None)
            try:
                percent = p.memory_percent()
            except Exception:
                percent = None
    except Exception:
        rss = vms = percent = None

    ru_maxrss = None
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        ru_maxrss = getattr(ru, 'ru_maxrss', None)
    except Exception:
        ru_maxrss = None

    return {
        'timestamp': ts,
        'rss': rss,
        'vms': vms,
        'percent': percent,
        'ru_maxrss': ru_maxrss,
    }


def append_memory_csv(csv_path, row):
    """Append a row (dict) to csv_path. Creates parent dir and header if needed."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    # normalize keys order: ensure timestamp first, then slide_id/patch_idx then others
    keys = list(row.keys())
    if write_header:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerow(row)
    else:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writerow(row)
