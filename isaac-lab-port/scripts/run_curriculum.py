"""Automatic static-obstacle curriculum.  Plain python (no Isaac imports); it launches train.py / eval_policy.py as subprocesses.

For each stage: train from the previous stage's chosen checkpoint, evaluate every --eval_every-th checkpoint on the stage's own
setting (deterministic, 1024 envs, evaluation line).  The first checkpoint that passes the gate (reach >= --reach, collision <= --col)
is confirmed with --confirm_rounds rounds; if it still passes, training is stopped and that checkpoint seeds the next stage.
If a stage hits --max_iters without passing, the curriculum stops and reports the best checkpoint seen.
"""
import argparse, glob, os, re, signal, subprocess, sys, time

p = argparse.ArgumentParser()
p.add_argument("--start_ckpt", default=None)   # None -> first stage trains from scratch
p.add_argument("--lr", type=float, default=None)
p.add_argument("--stages", default="80,160,250,350")
p.add_argument("--prefix", default="cur_s")
p.add_argument("--reach", type=float, default=0.90)
p.add_argument("--col", type=float, default=0.10)
p.add_argument("--max_iters", type=int, default=12000)
p.add_argument("--eval_every", type=int, default=1000)   # iterations between evaluated checkpoints
p.add_argument("--first_eval", type=int, default=2000)
p.add_argument("--confirm_rounds", type=int, default=3)
p.add_argument("--num_dynamic", type=int, default=0)     # 0 -> --no_dynamic
a = p.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
PAT = re.compile(r"reached goal \(at any time\): ([\d.]+)% \| collision: ([\d.]+)%")
child = None

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def dyn_args():
    return ["--no_dynamic"] if a.num_dynamic == 0 else ["--num_dynamic", str(a.num_dynamic)]

def evaluate(ckpt, n_static, rounds):
    out = subprocess.run([sys.executable, f"{ROOT}/scripts/eval_policy.py", "--checkpoint", ckpt, "--num_envs", "1024",
                          "--episodes", str(rounds), "--num_static", str(n_static)] + dyn_args(),
                         capture_output=True, text=True).stdout
    m = PAT.search(out)
    if not m:
        log("  eval produced no result line"); return None
    return float(m[1]) / 100, float(m[2]) / 100

def stop_child():
    global child
    if child and child.poll() is None:
        child.send_signal(signal.SIGINT)
        try: child.wait(timeout=120)
        except subprocess.TimeoutExpired: child.kill()

signal.signal(signal.SIGTERM, lambda *_: (stop_child(), sys.exit(1)))

ckpt = a.start_ckpt
for n in [int(s) for s in a.stages.split(",")]:
    name = f"{a.prefix}{n}"
    run_dir = f"{ROOT}/logs/navrl/{name}"
    if os.path.exists(run_dir):
        log(f"{run_dir} already exists - abort"); sys.exit(1)
    log(f"=== stage static={n} dynamic={a.num_dynamic} | resume {ckpt}")
    child = subprocess.Popen([sys.executable, f"{ROOT}/scripts/train.py"] + (["--resume", ckpt] if ckpt else []) + (["--lr", str(a.lr)] if a.lr else []) + ["--num_envs", "1024",
                              "--num_static", str(n), "--max_iterations", str(a.max_iters), "--save_interval", "500",
                              "--run_name", name] + dyn_args(),
                             stdout=open(f"{LOGS}/{name}.out", "w"), stderr=subprocess.STDOUT)
    open(f"{LOGS}/{name}.pid", "w").write(str(child.pid))
    done, best, chosen = set(), (-9, None, None), None
    while chosen is None:
        time.sleep(30)
        running = child.poll() is None
        for f in sorted(glob.glob(f"{run_dir}/checkpoint_[0-9]*.pt"), key=lambda s: int(re.findall(r"_(\d+)\.pt", s)[0])):
            it = int(re.findall(r"_(\d+)\.pt", f)[0])
            if it in done or it < a.first_eval or it % a.eval_every or time.time() - os.path.getmtime(f) < 15:
                continue
            done.add(it)
            r = evaluate(f, n, 1)
            if r is None: continue
            log(f"  it {it}: reach {r[0]:.1%} collision {r[1]:.1%}")
            if r[0] - r[1] > best[0]: best = (r[0] - r[1], it, f)
            if r[0] >= a.reach and r[1] <= a.col:
                c = evaluate(f, n, a.confirm_rounds)
                log(f"  confirm it {it} ({a.confirm_rounds} rounds): {c}")
                if c and c[0] >= a.reach and c[1] <= a.col:
                    chosen = f; stop_child(); break
        if chosen is None and not running:
            log(f"stage {n} ended without passing; best checkpoint {best[2]} (score {best[0]:.3f}). Curriculum stops here.")
            sys.exit(2)
    log(f"stage {n} PASSED with {chosen}")
    ckpt = chosen
log(f"curriculum finished; final checkpoint {ckpt}")
