# Source Of Truth

## Rule

GitHub is the source of truth for WorldExplorer logic.

Kaggle kernels are launch surfaces. They should not contain modeling logic,
search policy, feature-space operators, memory/atlas logic, fleet strategy, or
submission heuristics except as generated configuration.

The normal flow is:

```text
GitHub repo
  -> pip install in Kaggle when internet is enabled
  -> attached wheel/source fallback only for offline notebooks
  -> wx.kaggle.run(CONFIG)
  -> submission.csv + telemetry artifacts
```

## What Belongs In GitHub

These are source artifacts and must live in this repo:

```text
engine_src/
  Modular WorldExplorer engine sources.

worldexplorer/_engine.py
  Committed generated engine, rebuilt from engine_src by sync_engine.py.

worldexplorer/
  Package API, Kaggle adapter, CLI, autoconfig, run interface.

tools/fleet.py
  Fleet generation, breaker/grok/sprout run definitions, Kaggle metadata.

tools/memory_matrices.py
  Durable atlas/memory matrix extraction from run artifacts.

tools/telemetry_guidance.py
  Cross-run telemetry guidance and next-action recommendations.

tools/route_carve.py
  Submission-space diagnostics and reversible route-carve experiments.

tools/worldview_loop.py
  Runtime loop support for reviewing and feeding observations back in.

kaggle/bootstrap_kernel.py
  Slim bootstrap template only. It fetches the package and calls wx.kaggle.run.

examples/
  Small local and Kaggle entrypoint examples.

tests/
  Package and adapter tests.
```

## What Does Not Belong In Kaggle Kernels

A Kaggle kernel should not contain:

```text
full engine source
model families
feature transforms
search loops
memory atlas builders
telemetry analyzers
route-carve logic
blend/submit heuristics
large copied helper modules
```

Those belong in GitHub and are loaded as the `worldexplorer` package.

## What A Slim Kernel May Contain

A slim Kaggle kernel may contain only:

```text
CONFIG
package acquisition policy
import worldexplorer as wx
wx.kaggle.run(CONFIG)
```

The default acquisition policy is:

```text
source_policy = "github_first"
enable_internet = true
repo = git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git@master
```

For offline notebooks:

```text
source_policy = "wheel_first"
enable_internet = false
dataset_sources = ["<owner>/worldexplorer-engine"]
```

## Generated Artifacts

Generated fleet notebooks under a local workspace such as:

```text
/home/username/new_algo/kaggle/fleet/
```

are not source of truth. They are generated outputs from `tools/fleet.py`.

If a generated kernel contains a large copy of the engine, it is legacy or a
temporary artifact. The current preferred fleet path is slim GitHub-first
generation.

## Legacy Workspace Files

The top-level workspace may contain unrelated or older research logic, including
legacy competition scaffolds. For example:

```text
/home/username/new_algo/kernel.py
```

is a legacy ROGII scaffold, not the DRW WorldExplorer package source.

WorldExplorer source changes should be made in:

```text
/home/username/new_algo/worldexplorer
```

and pushed to:

```text
https://github.com/Amarel-Taylor-Scott/worldexplorer
```

## Enforcement

Run:

```bash
python tools/source_audit.py
```

The audit checks that:

```text
engine source exists in GitHub repo
generated engine exists in the package
bootstrap stays slim
bootstrap uses github_first by default
bootstrap calls wx.kaggle.run(CONFIG)
fleet defaults to GitHub-first master kernels
```

This is not a substitute for code review, but it catches the main regression:
accidentally moving logic back into a Kaggle notebook.
