"""Generate the saved AI analyses that power demo mode.

Runs the real pipeline (two model calls per snapshot: theme notes, then the briefing) with the
app's default settings and stores the result under demo/analyses/<fingerprint>.json. Existing
analyses are skipped unless --force is given. Needs OPENROUTER_API_KEY.

    python scripts/build_demo.py [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketpulse.citations import check_citations  # noqa: E402
from marketpulse.demo import load_saved_analysis, load_snapshots, save_analysis  # noqa: E402
from marketpulse.llm import LLMClient, LLMError  # noqa: E402
from marketpulse.pipeline import Params, compute, fingerprint, run_ai  # noqa: E402


def main(force: bool = False) -> int:
    llm = LLMClient(app_name="MarketPulse AI")
    params = Params()  # must match the app's default sidebar settings
    failures = 0
    for ds in load_snapshots():
        fp = fingerprint(ds, params)
        name = ds.watchlist.name
        if load_saved_analysis(fp) and not force:
            print(f"skip   {name} ({fp}): already saved")
            continue
        try:
            analysis = run_ai(llm, ds, params, compute(ds, params), on_step=lambda msg: print(f"       {msg}"))
        except LLMError as exc:
            failures += 1
            print(f"FAILED {name}: {exc}")
            continue
        check = check_citations(analysis["briefing"], len(analysis["references"]))
        path = save_analysis(analysis)
        print(f"saved  {name} -> {path.name} ({analysis['model']}); citation coverage {check.coverage:.0%}, "
              f"invalid refs {check.invalid_refs or 'none'}")
    print(f"model calls made: {llm.calls_made}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv))
