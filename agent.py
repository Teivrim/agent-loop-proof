"""A perpetual agent: run a pass, then hand off to the next run.

The 6-hour job cap is the reason a `while True` agent on a free runner is
pointless - the job dies and the wait is lost. But minutes on a *public*
repository are free and unmetered, so the cap stops being the limit: a job only
has to be shorter than the cap, and the work continues in the next run.

So the loop is moved out of the process and into the workflow. This script does
one pass, records what it learned, and exits with the verdict:

    0  finished - do not run again
    1  more work remains - run again

`.github/workflows/agent.yml` reads that exit code and calls
`gh workflow run` on itself, which is the whole mechanism. Nothing sleeps inside
a job, so a runner is never held open waiting for a quota window.

State lives in `agent_state.json` in the working tree and is committed back by
the workflow, which is what makes the agent's memory survive between runs. A
runner that starts from scratch would re-do work it already did - and for a
channel that means re-uploading episodes.

`stop` is a plain file. Touch it and the agent finishes its current pass and
exits 0; the next run is never requested. That is the off switch, and it is a
file rather than a variable because a workflow input cannot be read by the
process that has to decide whether to ask for the next run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "agent_state.json"
STOP = HERE / "stop"

# A pass that reports "more work" without the total moving is a pass that is
# not making progress. Two of those in a row is treated as done: the honest
# outcome of an agent that cannot advance is to stop, not to spend the rest of
# the month re-running the same no-op.
NO_PROGRESS_LIMIT = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                     encoding="utf-8")


def set_total(value) -> None:
    """Record what the work has produced so far.

    Called by the pass itself - a pass that finishes two of ten uploads sets
    `{"done": 2}` - and that value is what the next run compares against. The
    agent has no way of knowing how much work exists otherwise, so without this
    every pass looks identical and the stall counter stops the agent after two.
    """
    state = load()
    state["total"] = value
    save(state)


def run_pass(command: list[str]) -> int:
    """One pass. Streams nothing - the log has it; this prints the verdict."""
    if not command:
        return 0
    started = time.time()
    r = subprocess.run(command, cwd=str(HERE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    state = load()
    state["last_pass"] = {
        "command": command,
        "at": now(),
        "rc": r.returncode,
        "seconds": round(time.time() - started, 1),
        # The last three lines of stdout are the step's own summary, which is
        # what decides whether the pass moved anything.
        "summary": [x for x in (r.stdout or "").strip().splitlines()[-3:] if x.strip()],
    }
    if r.returncode != 0:
        # Same reasoning as channel_sync.run(): a traceback is longer than a
        # few lines, and the part that names the fault is at the end.
        state["last_pass"]["stderr"] = [x for x in
                                        (r.stderr or "").strip().splitlines()[-4:]
                                        if x.strip()]
    save(state)
    return r.returncode


def progress_key() -> str:
    """What counts as 'the work moved'."""
    return json.dumps(load().get("total"), sort_keys=True)


def decide(stalled: int, force: bool, rc: int) -> tuple[bool, str]:
    """The single place that answers 'run again?', and it always says why.

    Every caller has to print a verdict: the workflow reads AGENT_VERDICT to
    decide whether to call `gh workflow run`, and a run that exits without one
    is read as 'do not continue', which silently ends the agent.
    """
    if STOP.exists():
        return False, "stop file present - finishing"
    if force:
        return True, f"forced ({stalled} stalled pass(es) ignored)"
    if stalled >= NO_PROGRESS_LIMIT:
        return False, (f"{stalled} passes with no change in 'total' - stopping "
                       "rather than looping forever")
    return True, f"progress made ({stalled} stalled pass(es))"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pass-cmd", default="",
                    help="command for one pass. Empty means inspect only.")
    ap.add_argument("--total", default="",
                    help="JSON written to state['total'] before the pass; the "
                         "agent is finished when two passes leave it unchanged")
    ap.add_argument("--force", action="store_true",
                    help="keep going regardless of progress (a pure poller)")
    ap.add_argument("--print-state", action="store_true")
    # parse_known_args, not parse_args: everything after `--` is the pass
    # command, and parse_args treats those as its own and exits 2.
    args, rest = ap.parse_known_args(argv)
    # The separator itself comes back in `rest`, so it would be run as a command
    # named `--`. Drop it, and with it any leading separators a caller stacked.
    while rest and rest[0] == "--":
        rest = rest[1:]

    if args.print_state:
        print(json.dumps(load(), ensure_ascii=False, indent=1))
        return 0

    state = load()
    state["runs"] = state.get("runs", 0) + 1
    state["last_run_at"] = now()
    # Written before the pass, because run_pass() reloads the file from disk and
    # would otherwise save a state dict that never saw this run - the counter
    # then stayed at its old value and the final `state["runs"]` raised
    # KeyError. The run has happened whether or not the pass manages anything.
    save(state)

    # A stop file ends the agent, but it still has to say so. Returning early
    # without a verdict looked identical to "stopped because finished" to the
    # workflow and to a human reading the log.
    if STOP.exists():
        state = load()
        state["verdict"] = "stop file present - the agent is finished"
        save(state)
        print(f"  run {state['runs']}: {state['verdict']}")
        print("AGENT_VERDICT=done")
        return 0

    before = progress_key()
    expected = json.loads(args.total) if args.total else None
    if expected is not None:
        state = load()
        state["expected_total"] = expected
        save(state)

    rc = 0
    # shlex.split() is deliberately not used: it reads POSIX quoting, so a
    # nested Windows command in a workflow file dies with
    # `ValueError: No closing quotation` before the pass even starts. The pass
    # arrives as separate arguments after `--`, which no shell can reinterpret.
    if rest:
        command = rest
        print(f"pass: {' '.join(command)}", flush=True)
        rc = run_pass(command)
    elif args.pass_cmd:
        command = args.pass_cmd.split()
        print(f"pass: {args.pass_cmd}", flush=True)
        rc = run_pass(command)

    after = progress_key()
    state = load()
    stalled = state.get("stalled", 0)
    if after != before:
        stalled = 0
    elif state.get("last_pass") is not None:
        # No change, but only count it as a stall once a pass has actually run.
        # A run that inspected state and did nothing is not a failed attempt.
        stalled += 1
    state["stalled"] = stalled

    more, why = decide(stalled, args.force, rc)
    # Reload before writing: the pass may have set `total` through
    # --set-total, and this dict was read before it ran.
    state = load()
    state["stalled"] = stalled
    state["verdict"] = f"run {state['runs']}: {why}"
    save(state)

    print(f"\n  pass rc={rc}  stalled={stalled}")
    print(f"  {state['verdict']}")
    print("AGENT_VERDICT=" + ("more" if more else "done"))
    return 0


if __name__ == "__main__":
    # `agent.py --set-total '{"done": 7}'` is how a pass reports progress, and
    # it is a separate entry point so a pass can call it without re-entering the
    # run/verdict machinery it is being measured by.
    if "--set-total" in sys.argv:
        set_total(json.loads(sys.argv[sys.argv.index("--set-total") + 1]))
        print("total set to " + json.dumps(load().get("total")))
        raise SystemExit(0)
    raise SystemExit(main())
