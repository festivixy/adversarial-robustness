# Adversarial Robustness Gap

This project compares two estimates of the L-infinity robustness radius of small ReLU classifiers
on a two-dimensional dataset. A convex-relaxation certificate gives a lower bound L, the FAB attack
gives an upper bound U, and a mixed-integer solver gives a reference interval for the true radius.
The networks have one, two, or three hidden layers and about the same number of parameters.

## Setup

| Computer | Command |
|---|---|
| Windows | `scripts\windows\setup.bat` |
| Linux | `scripts/linux/setup.sh` |
| Raspberry Pi (64-bit OS) | `scripts/pi/setup.sh` |

Each script installs [uv](https://docs.astral.sh/uv/), Python 3.12, and the package versions in
`uv.lock`, then starts the lab app's setup. Tests: `uv run pytest`.

## Collecting data

Each trained network is one job. Each computer has a name such as `lab-07`.

With the coordinator: on the Raspberry Pi, run `scripts/pi/coordinator.sh configs/pilot.yaml`, or
`scripts/pi/install-service.sh configs/pilot.yaml` to start it at boot. It prints its address, an
access code, and a setup command. On each Windows lab computer, run that command in PowerShell:

```powershell
irm http://<pi-address>:8765/bootstrap.ps1 | iex
```

The command downloads the project from the Pi, installs it, and opens the lab app's setup with the
Pi's address filled in. The lab computers then take jobs from the Pi and upload results to it. The
Pi's status page lists all jobs and computers.

Without the coordinator: choose jobs by slot in the lab app (`scripts\windows\lab.bat`) and move
results between computers with Export bundle and Import bundle.

Commands:

```bash
uv run robustness-gap lab
uv run robustness-gap plan     --config configs/pilot.yaml --machines 22
uv run robustness-gap work     --config configs/pilot.yaml --machine lab-07 --slot 7/22
uv run robustness-gap server   --config configs/pilot.yaml
uv run robustness-gap worker
uv run robustness-gap progress --config configs/pilot.yaml
uv run robustness-gap analyze  --config configs/pilot.yaml
```

`worker` without options uses the settings saved by the lab app. `--override
configs/overrides/smoke.yaml` runs a short test whose outputs are named `smoke-*` and are not
tracked by Git. The main configuration runs only with `--main-run`, with every setting filled in,
and with a recorded protocol freeze whose hash matches.
