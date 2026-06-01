import csv, math
#============== Shcedule LR ==========================
def read_lr_schedule(csv_path):
    """Read CSV and return list of (start, end, lr_start, lr_end, type)."""
    segments = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(row["start_epoch"])
                e = int(row["end_epoch"])
                lr0 = float(row["start_lr"])
                lr1 = float(row["end_lr"])
                typ = row.get("type", "linear").strip().lower()
                if typ in ("cosene", "consene", "cos", "cosine_decay"):
                    typ = "cosine"
                segments.append((s, e, lr0, lr1, typ))
            except Exception as ex:
                print(f"⚠️ Skipping bad row: {row} ({ex})")
    return segments


def get_lr_from_segments(epoch, segments):
    """Return learning rate for the given epoch according to the segments list."""
    for (s, e, lr0, lr1, typ) in segments:
        if s <= epoch <= e:
            t = (epoch - s) / max(1, (e - s))
            if typ == "linear":
                return lr0 + t * (lr1 - lr0)
            elif typ == "cosine":
                return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * t))
            else:
                return lr0  # constant or unknown type
    # fallback before/after schedule
    if epoch < segments[0][0]:
        return segments[0][2]
    return segments[-1][3]
