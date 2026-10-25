from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as C
from . import collection, lab, records


def _load(args, guard: bool = True) -> tuple[dict, dict]:
    authored, cfg = C.load_config(args.config, args.override or [])
    if guard:
        C.check_main_run_allowed(cfg, getattr(args, "main_run", False))
    return authored, cfg


def write_manifest(root: Path, args, authored: dict, cfg: dict, command: str) -> Path:
    from .provenance import environment

    path = root / "runs" / cfg["experiment"]["id"] / "invocations" / f"{collection.stamp()}-{command}.json"
    records.atomic_write_json(path, {
        "command": command,
        "argv": sys.argv,
        "config_path": str(args.config),
        "overrides": [str(o) for o in (args.override or [])],
        "authored_config": authored,
        "scientific_hash": C.scientific_hash(cfg),
        "unresolved_settings": C.unresolved_settings(cfg),
        "environment": environment(root),
    })
    return path


def cmd_validate(args) -> int:
    from .models import check_architecture_table
    from .provenance import environment

    _, cfg = _load(args, guard=False)
    env = environment(Path(args.root))
    print(f"experiment:      {cfg['experiment']['id']} (stage {cfg['experiment']['stage']})")
    print(f"scientific hash: {C.scientific_hash(cfg)}")
    print(f"code hash:       {env['code_hash']}")
    problems = check_architecture_table(cfg["model"]["architectures"])
    print("architectures:   " + ("OK (182/36, 189/22, 186/24)" if not problems else f"MISMATCH {problems}"))
    unresolved = C.unresolved_settings(cfg)
    print(f"unresolved settings ({len(unresolved)}):")
    for u in unresolved:
        print(f"  - {u}")
    print(f"environment:     python {env['python']}, torch {env['packages']['torch']}, "
          f"scipy {env['packages']['scipy']}, {env['cpu_model']}, {env['logical_cpus']} logical CPUs")
    try:
        C.check_main_run_allowed(cfg, args.main_run)
        print("run guard:       OK for this stage")
    except C.ConfigError as e:
        print(f"run guard:       BLOCKED - {e}")
        return 1
    return 1 if problems else 0


def cmd_generate_data(args) -> int:
    from .data import generate_dataset, load_dataset
    from .plots import plot_dataset

    authored, cfg = _load(args)
    root = Path(args.root)
    path = generate_dataset(cfg, root / "data")
    figs = plot_dataset(load_dataset(path), root / "figures" / cfg["experiment"]["id"] / "dataset_train")
    write_manifest(root, args, authored, cfg, "generate-data")
    print(f"dataset: {path}")
    print(f"figure:  {figs[0]}")
    return 0


def print_plan(cfg: dict, machines: int | None) -> None:
    jobs = collection.all_jobs(cfg)
    print(f"{len(jobs)} jobs in experiment {cfg['experiment']['id']}:")
    for i, j in enumerate(jobs, 1):
        print(f"  {i:>3}. {j.job_id}")
    if machines:
        print(f"\nsplit across {machines} machines (machine K runs slot K/{machines}):")
        for k, js in collection.assign(jobs, machines).items():
            print(f"  slot {k:>2}/{machines}: {', '.join(j.job_id for j in js) or '(nothing)'}")


def cmd_plan(args) -> int:
    _, cfg = _load(args, guard=False)
    print_plan(cfg, args.machines)
    return 0


def run_work(root: Path, args, authored: dict, cfg: dict, jobs: list, machine: str, workers: int) -> int:
    write_manifest(root, args, authored, cfg, "work")
    print(f"machine {machine}: {len(jobs)} jobs, {workers} worker(s)")
    summaries = lab.work(root, cfg, jobs, machine, workers)
    failed = [s for s in summaries if "error" in s]
    print(f"finished: {len(summaries) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


def cmd_work(args) -> int:
    authored, cfg = _load(args)
    jobs = collection.select_jobs(cfg, args.job, args.slot, args.all)
    return run_work(Path(args.root), args, authored, cfg, jobs, args.machine, args.workers)


def print_progress(report: dict) -> None:
    print(f"panel size: {report['panel_size']}")
    print(f"{'job':<16} {'trained':<18} {'machine':<12} {'clean':>6} {'cert':>6} {'attack':>7} {'ref':>5}")
    for r in report["jobs"]:
        print(f"{r['job']:<16} {r['trained']:<18} {r['machine'] or '-':<12} {r['clean']:>6} "
              f"{r['certificate']:>6} {r['attack']:>7} {r['reference']:>5}")
    if report["machines"]:
        print(f"\n{'machine':<12} {'sessions':>8} {'notes':>6} {'tests':>6} {'fingerprint':<13} {'seconds':>8}  cpu")
        for name, m in report["machines"].items():
            tests = {True: "pass", False: "FAIL", None: "-"}[m["tests_passed"]]
            secs = f"{m['fingerprint_seconds']:.1f}" if m["fingerprint_seconds"] is not None else "-"
            print(f"{name:<12} {m['sessions']:>8} {m['notes']:>6} {tests:>6} {m['fingerprint'] or '-':<13} "
                  f"{secs:>8}  {m['cpu'] or '-'}")
        prints = {m["fingerprint"] for m in report["machines"].values() if m["fingerprint"]}
        if len(prints) > 1:
            print("\nmachines produced different fingerprints, so training is not bit-identical across them")


def cmd_progress(args) -> int:
    _, cfg = _load(args, guard=False)
    print_progress(collection.progress(Path(args.root), cfg))
    return 0


def print_selfcheck(r: dict) -> bool:
    t = r["tests"]
    status = "skipped" if not t["ran"] else ("pass" if t["passed"] else "FAIL")
    print(f"tests:       {status} {t.get('summary', '')}")
    print(f"fingerprint: {r['fingerprint']}  ({r['fingerprint_network']}, {r['fingerprint_seconds']:.1f}s)")
    print(f"cpu:         {r['environment']['cpu_model']} ({r['environment']['logical_cpus']} logical CPUs)")
    print(f"code hash:   {r['environment']['code_hash']}")
    return not t["ran"] or t["passed"]


def cmd_selfcheck(args) -> int:
    _, cfg = _load(args, guard=False)
    print(f"checking machine {args.machine} ...")
    return 0 if print_selfcheck(lab.selfcheck(Path(args.root), cfg, args.machine, not args.skip_tests)) else 1


def cmd_note(args) -> int:
    _, cfg = _load(args, guard=False)
    collection.add_note(Path(args.root), cfg, args.machine, " ".join(args.text))
    print("note saved")
    return 0


def print_notes(notes: list[dict]) -> None:
    for n in notes:
        print(f"{n['time']}  [{n['machine']}]  {n['note']}")
    if not notes:
        print("no notes yet")


def cmd_notes(args) -> int:
    _, cfg = _load(args, guard=False)
    print_notes(collection.all_notes(Path(args.root), cfg))
    return 0


def cmd_export(args) -> int:
    _, cfg = _load(args, guard=False)
    print(f"bundle written: {collection.export_bundle(Path(args.root), cfg, args.machine)}")
    return 0


def print_import(r: dict) -> None:
    print(f"from {r['from_machine']}: {r['added']} added, {r['identical']} already present, "
          f"{r['extended_logs']} logs extended, {len(r['conflicts_kept_existing'])} conflicts kept as existing")


def cmd_import(args) -> int:
    _, cfg = _load(args, guard=False)
    print_import(collection.import_bundle(Path(args.root), cfg, args.bundle, args.keep_existing, args.allow_code_mismatch))
    return 0


def print_analysis(s: dict) -> None:
    print(f"complete: {s['complete']}")
    if s["complete"]:
        print(f"inputs correct for every model: {s['common_correct_subset_size']} of {s['panel_size']}")
    for key in ("missing_models", "failed_models"):
        if s[key]:
            print(f"{key.replace('_', ' ')}: {', '.join(s[key])}")
    if s["missing_instances"]:
        print(f"missing instances: {len(s['missing_instances'])}")


def cmd_analyze(args) -> int:
    from .analyze import summarize_clean

    _, cfg = _load(args, guard=False)
    print_analysis(summarize_clean(Path(args.root), cfg))
    return 0


def cmd_lab(args) -> int:
    from .tui import run_app

    return run_app(Path(args.root), force_setup=args.setup, coordinator=args.coordinator)


def cmd_server(args) -> int:
    from .server import Coordinator, lan_address, make_server

    sys.stdout.reconfigure(line_buffering=True)
    _, cfg = _load(args)
    coordinator = Coordinator(Path(args.root), cfg, token=args.access_code, lease_seconds=args.lease,
                              reset_failed=args.reset_failed)
    server = make_server(coordinator, args.host, args.port)
    port = server.server_address[1]
    address = f"http://{lan_address()}:{port}"
    print(f"coordinator for {cfg['experiment']['id']} is running")
    print(f"  status page:   {address}/")
    print(f"  access code:   {coordinator.token}")
    print("  set up a Windows lab computer by pasting this into PowerShell:")
    print(f"    irm {address}/bootstrap.ps1 | iex")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        coordinator.events.write("coordinator_stop")
        print("\nstopped")
    return 0


def cmd_worker(args) -> int:
    from .settings import load_settings
    from .worker import CoordinatorClient, serve_jobs

    saved = load_settings(args.root)
    root = Path(args.root)
    config = args.config or (root / saved.config if saved else None)
    overrides = args.override or (saved.overrides(root) if saved and not args.config else [])
    machine = args.machine or (saved.machine if saved else None)
    url = args.coordinator or (saved.coordinator if saved else None)
    code = args.access_code or (saved.access_code if saved else None)
    missing = [n for n, v in (("--config", config), ("--machine", machine), ("--coordinator", url),
                              ("--access-code", code)) if not v]
    if missing:
        print(f"error: missing {', '.join(missing)} (or run the lab app setup first)", file=sys.stderr)
        return 1
    args.config, args.override = config, overrides
    authored, cfg = _load(args)
    write_manifest(root, args, authored, cfg, "worker")
    print(f"machine {machine} working for the coordinator at {url}")
    summaries = serve_jobs(root, cfg, CoordinatorClient(url, code), machine, args.workers, wait_for_more=args.wait)
    failed = [s for s in summaries if "error" in s]
    print(f"finished: {len(summaries) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="robustness-gap",
                                 description="The robustness-gap command and its subcommands.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def command(name, fn, help_text, machine=False):
        sp = sub.add_parser(name, help=help_text, description=help_text)
        sp.add_argument("--config", required=True, type=Path)
        sp.add_argument("--override", action="append", type=Path, help="smoke override file")
        sp.add_argument("--root", default=".", help="project folder (default: current folder)")
        if machine:
            sp.add_argument("--machine", required=True, help="short name of this computer, e.g. lab-07")
        sp.set_defaults(func=fn)
        return sp

    sp = command("validate", cmd_validate, "check the configuration and environment")
    sp.add_argument("--main-run", action="store_true")
    sp = command("generate-data", cmd_generate_data, "create or verify the dataset")
    sp.add_argument("--main-run", action="store_true")
    sp = command("plan", cmd_plan, "list the jobs and how they split across machines")
    sp.add_argument("--machines", type=int, help="number of computers to split the jobs across")
    sp = command("work", cmd_work, "run jobs on this computer", machine=True)
    which = sp.add_mutually_exclusive_group(required=True)
    which.add_argument("--job", action="append", help="a job id such as mlp2-seed3 (repeatable)")
    which.add_argument("--slot", help="this machine's share, e.g. 3/10")
    which.add_argument("--all", action="store_true", help="every job")
    sp.add_argument("--workers", type=int, default=1, help="jobs to run side by side (default 1)")
    sp.add_argument("--main-run", action="store_true")
    command("progress", cmd_progress, "show what has been done and where")
    sp = command("selfcheck", cmd_selfcheck, "test this computer before using it", machine=True)
    sp.add_argument("--skip-tests", action="store_true")
    sp = command("note", cmd_note, "write down an observation", machine=True)
    sp.add_argument("text", nargs="+")
    command("notes", cmd_notes, "print every note in time order")
    command("export", cmd_export, "pack this computer's results into a bundle", machine=True)
    sp = command("import", cmd_import, "merge a bundle from another computer")
    sp.add_argument("bundle", type=Path)
    sp.add_argument("--keep-existing", action="store_true", help="on conflict keep local files and continue")
    sp.add_argument("--allow-code-mismatch", action="store_true")
    command("analyze", cmd_analyze, "summarise the results present so far")
    sp = sub.add_parser("lab", help="open the lab app", description="open the lab app")
    sp.add_argument("--root", default=".", help="project folder (default: current folder)")
    sp.add_argument("--setup", action="store_true", help="run the setup walkthrough again")
    sp.add_argument("--coordinator", help="pre-fill the coordinator address in the walkthrough")
    sp.set_defaults(func=cmd_lab)
    sp = command("server", cmd_server, "run the coordinator (normally on the Raspberry Pi)")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--access-code", help="reuse a specific access code instead of the saved one")
    sp.add_argument("--lease", type=float, default=600, help="seconds before a silent computer loses its job")
    sp.add_argument("--reset-failed", action="store_true", help="put failed jobs back in the queue")
    sp.add_argument("--main-run", action="store_true")
    sp = sub.add_parser("worker", help="take jobs from the coordinator until none are left",
                        description="take jobs from the coordinator until none are left")
    sp.add_argument("--config", type=Path)
    sp.add_argument("--override", action="append", type=Path)
    sp.add_argument("--root", default=".")
    sp.add_argument("--machine")
    sp.add_argument("--coordinator", help="address of the coordinator, e.g. 192.168.1.50:8765")
    sp.add_argument("--access-code")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--wait", action="store_true", help="keep waiting when the queue is empty")
    sp.add_argument("--main-run", action="store_true")
    sp.set_defaults(func=cmd_worker)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (C.ConfigError, collection.LabError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
