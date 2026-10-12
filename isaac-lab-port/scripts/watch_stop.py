"""Stop a training run when it keeps getting worse.
score = reach - collision, averaged over a window of logged iterations (from the train log).
Stop when the latest window is worse than the best window by > --drop for --patience windows in a row."""
import argparse, os, re, signal, time
p = argparse.ArgumentParser()
p.add_argument("--log", required=True); p.add_argument("--pid", required=True)
p.add_argument("--window", type=int, default=500)    # iterations per window
p.add_argument("--drop", type=float, default=0.05)
p.add_argument("--patience", type=int, default=2)
p.add_argument("--min_iter", type=int, default=3000)
a = p.parse_args()
pat = re.compile(r"it\s+(\d+) .*collision ([\d.]+) \| reach ([\d.]+)")
pid = int(open(a.pid).read())
def alive():
    try: os.kill(pid, 0); return True
    except OSError: return False
best, bad, done_w = -9, 0, 0
while alive():
    time.sleep(120)
    rows = [(int(m[1]), float(m[3]) - float(m[2])) for l in open(a.log, errors="ignore") if (m := pat.search(l))]
    while rows and rows[-1][0] >= (done_w + 1) * a.window:
        w = [s for i, s in rows if done_w * a.window <= i < (done_w + 1) * a.window]
        sc = sum(w) / len(w); done_w += 1
        bad = bad + 1 if sc < best - a.drop else 0
        best = max(best, sc)
        print(f"window {done_w} (it<{done_w*a.window}) score {sc:.3f} best {best:.3f} bad {bad}", flush=True)
        if (done_w * a.window) >= a.min_iter and bad >= a.patience:
            print("degrading -> stopping training", flush=True)
            os.kill(pid, signal.SIGINT); raise SystemExit
print("training ended", flush=True)
