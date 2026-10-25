import asyncio
import shutil

import pytest
from textual.widgets import Button, Checkbox, DataTable, Input, RichLog, Select, TabbedContent

from robustness_gap.collection import all_notes

from robustness_gap.settings import LabSettings, load_settings, save_settings
from robustness_gap.tui import DashboardScreen, LabApp, SetupScreen

from conftest import REPO


@pytest.fixture
def project(tmp_path):
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    return tmp_path


def run(coro):
    return asyncio.run(coro)


async def wait_until(pilot, condition, timeout=120.0):
    waited = 0.0
    while not condition():
        await pilot.pause(0.2)
        waited += 0.2
        if waited > timeout:
            raise TimeoutError("condition not reached")


async def press(pilot, selector):
    pilot.app.screen.query_one(selector, Button).press()
    await pilot.pause(0.1)


def log_text(log: RichLog) -> str:
    return "\n".join(line.text for line in log.lines)


def test_first_launch_walks_through_setup(project):
    async def scenario():
        app = LabApp(project)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.5)
            assert isinstance(app.screen, SetupScreen)
            await press(pilot, "#next")
            app.screen.query_one("#smoke", Checkbox).value = True
            await press(pilot, "#next")
            app.screen.query_one("#machine-name", Input).value = "bad name"
            await press(pilot, "#next")
            assert app.screen.step == 2
            app.screen.query_one("#machine-name", Input).value = "lab-07"
            await press(pilot, "#next")
            await press(pilot, "#next")
            await press(pilot, "#next")
            assert app.screen.step == 4
            await press(pilot, "#check-run")
            await wait_until(pilot, lambda: app.screen.checked)
            await press(pilot, "#next")
            await press(pilot, "#next")
            await pilot.pause(0.5)
            assert isinstance(app.screen, DashboardScreen)

    run(scenario())
    saved = load_settings(project)
    assert saved.machine == "lab-07" and saved.smoke and saved.coordinator == ""


def test_dashboard_runs_jobs_and_records_notes(project):
    save_settings(project, LabSettings(config="configs/pilot.yaml", smoke=True, machine="lab-03"))

    async def scenario():
        app = LabApp(project)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = app.screen
            assert isinstance(screen, DashboardScreen)
            screen.query_one("#run-mode", Select).value = "all"
            await press(pilot, "#run-start")
            log = screen.query_one("#run-log", RichLog)
            await wait_until(pilot, lambda: "finished" in log_text(log))
            assert "3 ok, 0 failed" in log_text(log)
            await pilot.pause(0.5)
            assert screen.query_one("#data-table", DataTable).row_count == 12
            assert screen.query_one("#progress-table", DataTable).row_count == 3
            screen.query_one(TabbedContent).active = "tab-notes"
            await pilot.pause(0.2)
            screen.query_one("#note-text", Input).value = "fan was loud"
            await press(pilot, "#note-save")
            await pilot.pause(0.2)
            assert "fan was loud" in log_text(screen.query_one("#notes-log", RichLog))
            assert [n["note"] for n in all_notes(project, screen.cfg)] == ["fan was loud"]

    run(scenario())


def test_bad_slot_is_reported_not_crashing(project):
    save_settings(project, LabSettings(config="configs/pilot.yaml", smoke=True, machine="lab-03"))

    async def scenario():
        app = LabApp(project)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = app.screen
            screen.query_one("#run-mode", Select).value = "slot"
            screen.query_one("#run-which", Input).value = "9/4"
            await press(pilot, "#run-start")
            await pilot.pause(0.2)
            assert "between 1 and 4" in log_text(screen.query_one("#run-log", RichLog))
            assert not screen.query_one("#run-start", Button).disabled

    run(scenario())


def test_broken_saved_setup_falls_back_to_walkthrough(project):
    save_settings(project, LabSettings(config="configs/missing.yaml", machine="lab-03"))

    async def scenario():
        app = LabApp(project)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.2)
            assert isinstance(app.screen, SetupScreen)

    run(scenario())
