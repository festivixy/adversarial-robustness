from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

from rich.syntax import Syntax
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (Button, Checkbox, ContentSwitcher, DataTable, Footer, Header, Input,
                             Label, RichLog, Select, Static, TabbedContent, TabPane)

from . import collection
from .config import ConfigError, check_main_run_allowed, scientific_hash, unresolved_settings
from .settings import LabSettings, load_settings, save_settings

STEPS = ("welcome", "experiment", "machine", "coordinator", "check", "done")

WELCOME = """[b]Welcome to the lab app.[/b]

This computer will train small neural networks and measure how robust they are. Each trained
network is one [b]job[/b]. Jobs can be handed out by the coordinator (the Raspberry Pi), or you can
choose them yourself and carry the results over on a USB stick.

The next few steps set this computer up:

  1. choose the experiment
  2. give this computer a name
  3. connect to the coordinator, or skip to work offline
  4. run a self-check so we know this computer computes correctly

Everything you choose is saved, and you can change it later from the Machine tab.
"""

CSS = """
Screen { background: $surface; }
#wizard { padding: 1 2; height: 1fr; }
.step { height: auto; }
.title { text-style: bold; margin-bottom: 1; }
.hint { color: $text-muted; margin-bottom: 1; }
.row { height: auto; margin: 1 0; }
.row Input { width: 1fr; }
.row Button { margin-right: 1; }
#wizard-nav { height: auto; dock: bottom; padding: 1 2; }
#wizard-nav Button { margin-right: 2; }
#check-result, #coord-result, #experiment-info, #machine-error { height: auto; margin-top: 1; }
DataTable { height: 1fr; }
#run-log { height: 1fr; border: round $primary; }
#data-split { height: 1fr; }
#data-table { width: 3fr; }
#data-detail { width: 2fr; border: round $primary; padding: 0 1; }
#coordinator-table { height: 12; }
#data-filter { width: 44; }
#import-path { width: 1fr; }
#machine-log { height: 1fr; border: round $primary; }
#notes-log { height: 1fr; border: round $primary; }
#machine-info { height: auto; border: round $primary; padding: 0 1; margin-bottom: 1; }
.controls { height: auto; margin-bottom: 1; }
.controls Button, .controls Input, .controls Select { margin-right: 1; }
.controls Input { width: 24; }
#run-mode { width: 40; }
"""


def config_choices(root: Path) -> list[tuple[str, str]]:
    return [(p.name, p.relative_to(root).as_posix()) for p in sorted((root / "configs").glob("*.yaml"))]


class SetupScreen(Screen):
    BINDINGS = [Binding("escape", "app.quit", "Quit")]

    def __init__(self, root: Path, settings: LabSettings):
        super().__init__()
        self.root = root
        self.settings = settings
        self.step = 0
        self.checked = False
        self.title = "Lab setup"
        self.sub_title = f"step 1 of {len(STEPS)}"

    def compose(self) -> ComposeResult:
        yield Header()
        with ContentSwitcher(initial="welcome", id="wizard"):
            with VerticalScroll(id="welcome", classes="step"):
                yield Static(WELCOME)
            with VerticalScroll(id="experiment", classes="step"):
                yield Label("Which experiment?", classes="title")
                yield Static("The pilot is for trying things out. The main experiment is the real data "
                             "collection and only runs once the protocol is frozen.", classes="hint")
                choices = config_choices(self.root)
                current = self.settings.config if self.settings.config in [v for _, v in choices] else None
                yield Select(choices, value=current or (choices[0][1] if choices else Select.BLANK),
                             allow_blank=not choices, id="config-select")
                yield Checkbox("smoke test (a quick 3-epoch check whose results are never used)",
                               value=self.settings.smoke, id="smoke")
                yield Static(id="experiment-info")
            with VerticalScroll(id="machine", classes="step"):
                yield Label("What is this computer called?", classes="title")
                yield Static("Use the same name every time you use this computer, for example lab-07. "
                             "Letters, digits, - and _ only.", classes="hint")
                default = self.settings.machine or collection.suggest_machine_name(socket.gethostname())
                yield Input(value=default, id="machine-name")
                yield Static(id="machine-error")
            with VerticalScroll(id="coordinator", classes="step"):
                yield Label("Coordinator (optional)", classes="title")
                yield Static("If the Raspberry Pi coordinator is running, type its address and the access "
                             "code it printed. Leave both empty to work offline.", classes="hint")
                with Horizontal(classes="row"):
                    yield Input(value=self.settings.coordinator, placeholder="address, e.g. 192.168.1.50:8765",
                                id="coord-url")
                    yield Input(value=self.settings.access_code, placeholder="access code", id="coord-code")
                with Horizontal(classes="row"):
                    yield Button("Test connection", id="coord-test")
                yield Static(id="coord-result")
            with VerticalScroll(id="check", classes="step"):
                yield Label("Self-check", classes="title")
                yield Static("This runs the test suite and trains one tiny network to fingerprint this "
                             "computer. It takes about a minute.", classes="hint")
                yield Button("Run self-check", id="check-run", variant="primary")
                yield Static(id="check-result")
            with VerticalScroll(id="done", classes="step"):
                yield Label("All set", classes="title")
                yield Static(id="done-summary")
        with Horizontal(id="wizard-nav"):
            yield Button("Back", id="back")
            yield Button("Next", id="next", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.show_experiment_info()
        self.update_nav()

    def update_nav(self) -> None:
        self.query_one(ContentSwitcher).current = STEPS[self.step]
        self.sub_title = f"step {self.step + 1} of {len(STEPS)}"
        self.query_one("#back", Button).disabled = self.step == 0
        self.query_one("#next", Button).label = "Start" if STEPS[self.step] == "done" else "Next"

    def collect(self) -> LabSettings:
        value = self.query_one("#config-select", Select).value
        return LabSettings(
            config=value if isinstance(value, str) else self.settings.config,
            smoke=self.query_one("#smoke", Checkbox).value,
            machine=self.query_one("#machine-name", Input).value.strip(),
            coordinator=self.query_one("#coord-url", Input).value.strip(),
            access_code=self.query_one("#coord-code", Input).value.strip(),
            workers=self.settings.workers,
        )

    def show_experiment_info(self) -> None:
        info = self.query_one("#experiment-info", Static)
        try:
            _, cfg = self.collect().load_config(self.root)
        except (ConfigError, OSError, KeyError) as e:
            info.update(f"[red]cannot read this configuration: {e}[/red]")
            return
        jobs = collection.all_jobs(cfg)
        open_items = unresolved_settings(cfg)
        info.update(f"experiment [b]{cfg['experiment']['id']}[/b] (stage {cfg['experiment']['stage']}), "
                    f"{len(jobs)} jobs, {len(open_items)} undecided settings")

    @on(Select.Changed, "#config-select")
    @on(Checkbox.Changed, "#smoke")
    def experiment_changed(self) -> None:
        self.show_experiment_info()

    def validate_step(self) -> bool:
        name = STEPS[self.step]
        s = self.collect()
        if name == "experiment":
            try:
                s.load_config(self.root)
            except (ConfigError, OSError, KeyError) as e:
                self.notify(f"cannot use this configuration: {e}", severity="error")
                return False
        if name == "machine":
            try:
                collection.check_machine(s.machine)
            except collection.LabError as e:
                self.query_one("#machine-error", Static).update(f"[red]{e}[/red]")
                return False
            self.query_one("#machine-error", Static).update("")
        if name == "coordinator" and bool(s.coordinator) != bool(s.access_code):
            self.notify("type both the address and the access code, or leave both empty", severity="error")
            return False
        if name == "check" and not self.checked:
            self.notify("run the self-check first", severity="warning")
            return False
        return True

    @on(Button.Pressed, "#back")
    def back(self) -> None:
        self.step = max(0, self.step - 1)
        self.update_nav()

    @on(Button.Pressed, "#next")
    def next(self) -> None:
        if not self.validate_step():
            return
        if STEPS[self.step] == "done":
            self.settings = self.collect()
            save_settings(self.root, self.settings)
            self.app.open_dashboard(self.settings)
            return
        self.step += 1
        if STEPS[self.step] == "done":
            s = self.collect()
            where = f"coordinator at {s.coordinator}" if s.coordinator else "offline (no coordinator)"
            self.query_one("#done-summary", Static).update(
                f"This computer is [b]{s.machine}[/b], running [b]{s.config}[/b]"
                f"{' as a smoke test' if s.smoke else ''}, working {where}.\n\n"
                "Press Start to open the dashboard.")
        self.update_nav()

    @on(Button.Pressed, "#coord-test")
    def test_coordinator(self) -> None:
        s = self.collect()
        out = self.query_one("#coord-result", Static)
        if not s.coordinator:
            out.update("no address given; this computer will work offline")
            return
        out.update("connecting ...")

        def task() -> None:
            from .lab import machine_provenance
            from .worker import CoordinatorClient, RemoteError, check_compatible

            try:
                _, cfg = s.load_config(self.root)
                status = check_compatible(CoordinatorClient(s.coordinator, s.access_code, timeout=10), cfg,
                                          machine_provenance(self.root, s.machine or "check"))
                c = status["counts"]
                text = (f"[green]connected[/green] to experiment {status['experiment_id']}: "
                        f"{c['done']} done, {c['claimed']} running, {c['pending']} waiting")
            except (RemoteError, ConfigError, OSError) as e:
                text = f"[red]{e}[/red]"
            self.app.call_from_thread(out.update, text)

        self.run_worker(task, thread=True, group="coord-test", exclusive=True)

    @on(Button.Pressed, "#check-run")
    def run_check(self) -> None:
        s = self.collect()
        out = self.query_one("#check-result", Static)
        button = self.query_one("#check-run", Button)
        button.disabled = True
        out.update("running the self-check ...")

        def task() -> None:
            from .lab import selfcheck

            try:
                _, cfg = s.load_config(self.root)
                r = selfcheck(self.root, cfg, s.machine)
                self.checked = True
                text = describe_selfcheck(r)
            except Exception as e:
                text = f"[red]self-check failed: {type(e).__name__}: {e}[/red]"
            self.app.call_from_thread(out.update, text)
            self.app.call_from_thread(setattr, button, "disabled", False)

        self.run_worker(task, thread=True, group="selfcheck", exclusive=True)


def describe_selfcheck(r: dict | None) -> str:
    if not r:
        return "no self-check yet"
    t = r["tests"]
    tests = "skipped (no tests folder)" if not t["ran"] else (
        "[green]passed[/green]" if t["passed"] else "[red]FAILED[/red]")
    env = r["environment"]
    return (f"tests: {tests} {t.get('summary', '')}\n"
            f"fingerprint: {r['fingerprint'][:16] if r['fingerprint'] else '-'} "
            f"({r['fingerprint_network']}, {r['fingerprint_seconds']:.1f}s)\n"
            f"cpu: {env['cpu_model']} ({env['logical_cpus']} logical CPUs)\n"
            f"code hash: {env['code_hash'][:16]}   checked at {r['time']}")


class DashboardScreen(Screen):
    BINDINGS = [Binding("r", "refresh", "Refresh"), Binding("q", "request_quit", "Quit")]

    def __init__(self, root: Path, settings: LabSettings):
        super().__init__()
        self.root = root
        self.settings = settings
        self.authored, self.cfg = settings.load_config(root)
        self.stop_event = threading.Event()
        self.running = False
        self.data_rows: dict[str, dict] = {}
        exp = self.cfg["experiment"]
        self.title = f"{exp['id']} ({exp['stage']})"
        where = f" -> {settings.coordinator}" if settings.coordinator else " offline"
        self.sub_title = f"{settings.machine}{where}"

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-run"):
            with TabPane("Run", id="tab-run"):
                with Horizontal(classes="controls"):
                    modes = [("ask the coordinator for jobs", "coordinator"), ("my slot (e.g. 3/10)", "slot"),
                             ("specific jobs", "jobs"), ("all jobs", "all")]
                    default = "coordinator" if self.settings.coordinator else "slot"
                    yield Select(modes, value=default, allow_blank=False, id="run-mode")
                    yield Input(placeholder="slot or job ids", id="run-which")
                    yield Input(value=str(self.settings.workers), placeholder="jobs at once", id="run-workers")
                    yield Button("Start", id="run-start", variant="success")
                    yield Button("Stop", id="run-stop", variant="error", disabled=True)
                yield RichLog(id="run-log", wrap=True, markup=True)
            with TabPane("Progress", id="tab-progress"):
                with Horizontal(classes="controls"):
                    yield Button("Refresh", id="progress-refresh")
                    yield Button("Summarise results", id="progress-analyze")
                    yield Static(id="progress-summary")
                yield DataTable(id="progress-table", zebra_stripes=True)
                if self.settings.coordinator:
                    yield Label("Coordinator queue", classes="title")
                    yield DataTable(id="coordinator-table", zebra_stripes=True)
            with TabPane("Data", id="tab-data"):
                with Horizontal(classes="controls"):
                    yield Input(placeholder="filter by job, e.g. mlp2", id="data-filter")
                    yield Static(id="data-count")
                with Horizontal(id="data-split"):
                    yield DataTable(id="data-table", zebra_stripes=True, cursor_type="row")
                    with VerticalScroll(id="data-detail"):
                        yield Static("select a row to see its saved records", id="data-record")
            with TabPane("Notes", id="tab-notes"):
                with Horizontal(classes="controls"):
                    yield Input(placeholder="what happened? e.g. lab-07 rebooted at 10:40", id="note-text")
                    yield Button("Save note", id="note-save", variant="primary")
                yield RichLog(id="notes-log", wrap=True, markup=True)
            with TabPane("Machine", id="tab-machine"):
                yield Static(id="machine-info")
                with Horizontal(classes="controls"):
                    yield Button("Run self-check", id="machine-check")
                    yield Button("Export bundle", id="machine-export")
                    yield Button("Change setup", id="machine-setup")
                with Horizontal(classes="controls"):
                    yield Input(placeholder="path to a bundle .zip from another computer", id="import-path")
                    yield Button("Import bundle", id="machine-import")
                yield RichLog(id="machine-log", wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        progress = self.query_one("#progress-table", DataTable)
        progress.add_columns("job", "trained", "machine", "val acc", "clean", "cert", "attack", "ref")
        if self.settings.coordinator:
            self.query_one("#coordinator-table", DataTable).add_columns("job", "state", "machine", "attempts", "error")
        self.query_one("#data-table", DataTable).add_columns(
            "job", "input", "x1", "x2", "label", "pred", "margin", "correct", "machine", "later stages")
        self.query_one("#run-log", RichLog).write(
            f"[b]{self.settings.machine}[/b] is ready. Choose how to pick jobs and press Start.")
        self.action_refresh()
        self.set_interval(15, self.action_refresh)

    def say(self, text: str) -> None:
        self.app.call_from_thread(self.query_one("#run-log", RichLog).write, text)

    def action_refresh(self) -> None:
        self.refresh_progress()
        self.refresh_data()
        self.refresh_notes()
        self.refresh_machine()
        if self.settings.coordinator:
            self.run_worker(self.fetch_coordinator, thread=True, group="coord-status", exclusive=True)

    def refresh_progress(self) -> None:
        table = self.query_one("#progress-table", DataTable)
        table.clear()
        report = collection.progress(self.root, self.cfg)
        for r in report["jobs"]:
            acc = r["validation_accuracy"]
            table.add_row(r["job"], r["trained"], r["machine"] or "-", f"{acc:.3f}" if acc is not None else "-",
                          f"{r['clean']}/{report['panel_size']}", r["certificate"], r["attack"], r["reference"])
        trained = sum(r["trained"] == "success" for r in report["jobs"])
        self.query_one("#progress-summary", Static).update(f"  {trained} of {len(report['jobs'])} jobs trained here")

    def fetch_coordinator(self) -> None:
        from .worker import CoordinatorClient, RemoteError

        try:
            status = CoordinatorClient(self.settings.coordinator, self.settings.access_code, timeout=10).status()
        except RemoteError as e:
            status = {"error": str(e)}
        self.app.call_from_thread(self.show_coordinator, status)

    def show_coordinator(self, status: dict) -> None:
        table = self.query_one("#coordinator-table", DataTable)
        table.clear()
        if "error" in status:
            table.add_row("-", "unreachable", "-", "-", status["error"][:80])
            return
        for job_id, j in status["jobs"].items():
            table.add_row(job_id, j["state"], j["machine"] or "-", str(j["attempts"]), j["error"] or "")

    def refresh_data(self) -> None:
        table = self.query_one("#data-table", DataTable)
        needle = self.query_one("#data-filter", Input).value.strip()
        table.clear()
        self.data_rows = {}
        rows = [r for r in collection.instance_rows(self.root, self.cfg) if needle in r["job"]]
        for r in rows:
            key = f"{r['job']}/{r['sample_id']}"
            self.data_rows[key] = r
            later = ", ".join(st for st, ok in r["stages"].items() if ok) or "none yet"
            table.add_row(r["job"], r["sample_id"], f"{r['x'][0]:.3f}", f"{r['x'][1]:.3f}", str(r["true_label"]),
                          str(r["predicted_label"]), f"{r['clean_margin']:.4f}", "yes" if r["correct"] else "NO",
                          r["machine"], later, key=key)
        self.query_one("#data-count", Static).update(f"  {len(rows)} records")

    @on(Input.Changed, "#data-filter")
    def filter_changed(self) -> None:
        self.refresh_data()

    @on(DataTable.RowHighlighted, "#data-table")
    def show_record(self, event: DataTable.RowHighlighted) -> None:
        row = self.data_rows.get(event.row_key.value) if event.row_key else None
        if row is None:
            return
        parts = {}
        for path in sorted(row["folder"].glob("*.json")):
            parts[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(parts, indent=2)
        self.query_one("#data-record", Static).update(Syntax(text, "json", word_wrap=True))

    def refresh_notes(self) -> None:
        log = self.query_one("#notes-log", RichLog)
        log.clear()
        notes = collection.all_notes(self.root, self.cfg)
        for n in notes:
            log.write(f"[dim]{n['time']}[/dim]  [b]{n['machine']}[/b]  {n['note']}")
        if not notes:
            log.write("[dim]no notes yet[/dim]")

    def refresh_machine(self) -> None:
        s = self.settings
        latest = collection.latest_selfcheck(collection.Paths(self.root, self.cfg).machine_dir(s.machine))
        open_items = unresolved_settings(self.cfg)
        self.query_one("#machine-info", Static).update(
            f"machine [b]{s.machine}[/b]   config {s.config}{' (smoke)' if s.smoke else ''}   "
            f"coordinator {s.coordinator or 'none (offline)'}\n"
            f"settings hash {scientific_hash(self.cfg)[:16]}   undecided settings: {len(open_items)}\n\n"
            f"{describe_selfcheck(latest)}")

    @on(Button.Pressed, "#progress-refresh")
    def press_refresh(self) -> None:
        self.action_refresh()

    @on(Button.Pressed, "#progress-analyze")
    def press_analyze(self) -> None:
        from .analyze import summarize_clean

        try:
            s = summarize_clean(self.root, self.cfg)
        except FileNotFoundError as e:
            self.notify(str(e), severity="warning")
            return
        if s["complete"]:
            self.notify(f"{s['common_correct_subset_size']} of {s['panel_size']} inputs are correct for every model")
        else:
            self.notify(f"incomplete: {len(s['missing_models'])} models and "
                        f"{len(s['missing_instances'])} inputs missing", severity="warning")

    def chosen_jobs(self) -> list | None:
        mode = self.query_one("#run-mode", Select).value
        which = self.query_one("#run-which", Input).value.strip()
        if mode == "coordinator":
            return None
        if mode == "all":
            return collection.select_jobs(self.cfg, everything=True)
        if mode == "slot":
            return collection.select_jobs(self.cfg, slot=which)
        return collection.select_jobs(self.cfg, job_ids=[j.strip() for j in which.replace(",", " ").split()])

    @on(Button.Pressed, "#run-start")
    def start_run(self) -> None:
        log = self.query_one("#run-log", RichLog)
        try:
            check_main_run_allowed(self.cfg, self.cfg["experiment"]["stage"] == "main")
            workers = int(self.query_one("#run-workers", Input).value or "1")
            if workers < 1:
                raise ValueError("jobs at once must be at least 1")
            jobs = self.chosen_jobs()
            if jobs is None and not self.settings.coordinator:
                raise collection.LabError("no coordinator is set up; choose a slot or jobs, or change the setup")
        except (collection.LabError, ConfigError, ValueError) as e:
            log.write(f"[red]{e}[/red]")
            return
        self.stop_event.clear()
        self.set_running(True)
        log.write(f"[b]starting[/b] with {workers} job(s) at once")

        def task() -> None:
            from .cli import write_manifest
            from .lab import work
            from .worker import CoordinatorClient, serve_jobs

            try:
                write_manifest(self.root, ManifestArgs(self.root, self.settings), self.authored, self.cfg, "tui")
                if jobs is None:
                    client = CoordinatorClient(self.settings.coordinator, self.settings.access_code)
                    summaries = serve_jobs(self.root, self.cfg, client, self.settings.machine, workers,
                                           say=self.say, should_stop=self.stop_event.is_set)
                else:
                    summaries = work(self.root, self.cfg, jobs, self.settings.machine, workers,
                                     say=self.say, should_stop=self.stop_event.is_set)
                failed = sum("error" in s for s in summaries)
                self.say(f"[b]finished[/b]: {len(summaries) - failed} ok, {failed} failed")
            except Exception as e:
                self.say(f"[red]{type(e).__name__}: {e}[/red]")
            finally:
                self.app.call_from_thread(self.set_running, False)
                self.app.call_from_thread(self.action_refresh)

        self.run_worker(task, thread=True, group="jobs", exclusive=True)

    def set_running(self, running: bool) -> None:
        self.running = running
        self.query_one("#run-start", Button).disabled = running
        self.query_one("#run-stop", Button).disabled = not running

    @on(Button.Pressed, "#run-stop")
    def stop_run(self) -> None:
        self.stop_event.set()
        self.query_one("#run-log", RichLog).write("[yellow]stopping: running jobs will finish, no new ones start[/yellow]")

    @on(Button.Pressed, "#note-save")
    @on(Input.Submitted, "#note-text")
    def save_note(self) -> None:
        box = self.query_one("#note-text", Input)
        try:
            collection.add_note(self.root, self.cfg, self.settings.machine, box.value)
        except collection.LabError as e:
            self.notify(str(e), severity="warning")
            return
        box.value = ""
        self.refresh_notes()
        self.notify("note saved")

    @on(Button.Pressed, "#machine-check")
    def machine_check(self) -> None:
        log = self.query_one("#machine-log", RichLog)
        log.write("running the self-check ...")

        def task() -> None:
            from .lab import selfcheck

            try:
                text = describe_selfcheck(selfcheck(self.root, self.cfg, self.settings.machine))
            except Exception as e:
                text = f"[red]{type(e).__name__}: {e}[/red]"
            self.app.call_from_thread(log.write, text)
            self.app.call_from_thread(self.refresh_machine)

        self.run_worker(task, thread=True, group="selfcheck", exclusive=True)

    @on(Button.Pressed, "#machine-export")
    def machine_export(self) -> None:
        log = self.query_one("#machine-log", RichLog)
        try:
            log.write(f"bundle written: {collection.export_bundle(self.root, self.cfg, self.settings.machine)}")
        except collection.LabError as e:
            log.write(f"[red]{e}[/red]")

    @on(Button.Pressed, "#machine-import")
    def machine_import(self) -> None:
        log = self.query_one("#machine-log", RichLog)
        path = Path(self.query_one("#import-path", Input).value.strip().strip('"'))
        try:
            r = collection.import_bundle(self.root, self.cfg, path)
        except collection.ImportConflict as e:
            log.write(f"[red]{e}[/red]\nnothing was imported. If the files here should be kept, "
                      "use the command line: robustness-gap import --keep-existing")
            return
        except (collection.LabError, OSError) as e:
            log.write(f"[red]{e}[/red]")
            return
        log.write(f"from {r['from_machine']}: {r['added']} added, {r['identical']} already here, "
                  f"{r['extended_logs']} logs extended")
        self.action_refresh()

    @on(Button.Pressed, "#machine-setup")
    def machine_setup(self) -> None:
        if self.running:
            self.notify("stop the running jobs first", severity="warning")
            return
        self.app.open_setup(self.settings)

    def action_request_quit(self) -> None:
        if self.running and not self.stop_event.is_set():
            self.notify("jobs are running; press Stop first, then quit once they finish", severity="warning")
            return
        self.app.exit()


class ManifestArgs:
    def __init__(self, root: Path, settings: LabSettings):
        self.config = root / settings.config
        self.override = settings.overrides(root)


class LabApp(App):
    CSS = CSS
    TITLE = "Lab app"

    def __init__(self, root: str | Path, settings: LabSettings | None = None, force_setup: bool = False):
        super().__init__()
        self.root = Path(root)
        self.initial = settings or load_settings(self.root)
        self.force_setup = force_setup

    def on_mount(self) -> None:
        if self.initial is None or self.force_setup:
            self.open_setup(self.initial or LabSettings())
        else:
            try:
                self.open_dashboard(self.initial)
            except (ConfigError, OSError, KeyError) as e:
                self.notify(f"saved setup no longer works ({e}); please set up again", severity="error")
                self.open_setup(self.initial)

    def show(self, screen: Screen) -> None:
        if len(self.screen_stack) > 1:
            self.switch_screen(screen)
        else:
            self.push_screen(screen)

    def open_setup(self, settings: LabSettings) -> None:
        self.show(SetupScreen(self.root, settings))

    def open_dashboard(self, settings: LabSettings) -> None:
        self.show(DashboardScreen(self.root, settings))


def run_app(root: str | Path, force_setup: bool = False, coordinator: str | None = None) -> int:
    settings = load_settings(root)
    if coordinator:
        settings = settings or LabSettings()
        settings.coordinator = coordinator
        force_setup = True
    LabApp(root, settings, force_setup=force_setup).run()
    return 0
