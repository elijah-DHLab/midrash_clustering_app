"""Active working time per project, from Claude Code session logs.

    python tools/worktime.py --since 2026-10-19 [--until 2026-11-19] [--gap 15]


Active time: consecutive events in a session closer than GAP minutes count the gap between
them; a longer gap is a break. Each stretch is attributed to a project by the paths and
words in the events (tool inputs, user text); a stretch with no marker takes the last
project seen in that session.
"""
import collections
import datetime as dt
import glob
import json
import os
import re

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--since", required=True)
_ap.add_argument("--until", default=None)
_ap.add_argument("--gap", type=int, default=15, help="minutes; a longer pause is a break")
_args = _ap.parse_args()
GAP = dt.timedelta(minutes=_args.gap)
SINCE = dt.datetime.fromisoformat(_args.since).replace(tzinfo=dt.timezone.utc)
UNTIL = (dt.datetime.fromisoformat(_args.until).replace(tzinfo=dt.timezone.utc)
         if _args.until else dt.datetime.max.replace(tzinfo=dt.timezone.utc))
ROOTS = [os.path.expanduser("~/.claude/projects/c--Users-user-Code-Haifa"),
         os.path.expanduser("~/.claude/projects/C--Users-user-Code-Haifa-PHD"),
         os.path.expanduser("~/.claude/projects/c--Users-user-Code")]
MARKERS = [
    ("מדרש", re.compile(r"Midrash|midrash|מדרש|embedding_store|corpus_v7|sefaria|act_jsonl|DictaLM|סייר הקלאסטרים", re.I)),
    ("HAMATATE", re.compile(r"HAMATATE|hamatate|המטאטא|glossary|hamatate_tagging", re.I)),
    ("דוקטורט (poetic-decoupling)", re.compile(r"poetic-decoupling|poetic_decoupling|Code[\\/]+Haifa[\\/]+PHD|Overleaf|surprisal|jakobson", re.I)),
    ("מחשב ביתי", re.compile(r"ssh home|home_pc|id_ed25519_home|אורח", re.I)),
]


def label_of(text):
    hits = [name for name, rx in MARKERS if rx.search(text)]
    return hits[0] if hits else None


def events(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ts = e.get("timestamp")
        if not ts or e.get("type") not in ("user", "assistant"):
            continue
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if t < SINCE or t >= UNTIL:
            continue
        msg = e.get("message") or {}
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif c.get("type") == "tool_use":
                    parts.append(json.dumps(c.get("input", {}), ensure_ascii=False)[:3000])
        yield t, label_of(" ".join(parts)[:20000])


per_project = collections.Counter()
per_day = collections.defaultdict(collections.Counter)
sessions = collections.defaultdict(set)
for root in ROOTS:
    for path in glob.glob(os.path.join(root, "*.jsonl")):
        evs = sorted(events(path))
        if not evs:
            continue
        # fill unlabelled events from the nearest labelled one before (else after)
        labels = [l for _, l in evs]
        last = next((l for l in labels if l), "אחר")
        filled = []
        for l in labels:
            last = l or last
            filled.append(last)
        for (t0, _), (t1, _), lab in zip(evs, evs[1:], filled):
            gap = t1 - t0
            if gap <= GAP:
                per_project[lab] += gap.total_seconds()
                day = t0.astimezone(dt.timezone(dt.timedelta(hours=3))).date()
                per_day[day][lab] += gap.total_seconds()
                sessions[lab].add(os.path.basename(path))

print("since", SINCE.date(), "| gap threshold", GAP)
for lab, s in per_project.most_common():
    print(f"{lab:30} {s/3600:6.1f} h   sessions {len(sessions[lab])}")
print()
print("Midrash by day:")
tot = 0
for day in sorted(per_day):
    h = per_day[day]["מדרש"] / 3600
    if h >= 0.05:
        tot += h
        print(day, f"{h:5.1f} h")
print("total", round(tot, 1))
