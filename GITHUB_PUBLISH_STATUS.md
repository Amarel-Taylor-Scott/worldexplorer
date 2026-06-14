# GitHub Publish Status

## Short Answer

I can publish to GitHub from this environment.

The WorldExplorer package has already been published to:

```text
https://github.com/Amarel-Taylor-Scott/worldexplorer
```

The current published tag is:

```text
v0.2.2 -> 8191703bffced7c58416e9bb002fb80ed2d409db
```

## What Went Wrong In The Last Check

The failed Git commands were run from:

```text
/home/username/new_algo
```

That top-level directory contains an empty or invalid `.git` directory:

```text
/home/username/new_algo/.git
```

Because that `.git` directory is not a valid Git repository, `git status`,
`git remote`, and `git log` from the top-level workspace can fail or behave
incorrectly.

The actual publishable Git repository is nested here:

```text
/home/username/new_algo/worldexplorer
```

That repository is valid and has this remote:

```text
origin https://github.com/Amarel-Taylor-Scott/worldexplorer.git
```

## The Real Constraint

The constraint is not GitHub publishing.

The real constraint is Kaggle runtime access:

1. GitHub is the source of truth for code.
2. Kaggle competition kernels often run with internet disabled.
3. An internet-enabled Kaggle kernel should pull the current package directly
   from GitHub.
4. An internet-disabled Kaggle kernel cannot reliably `pip install` directly
   from GitHub during execution, so those notebooks need the attached wheel or
   source dataset fallback.

## Current Implemented Setup

The intended structure is:

```text
GitHub repo
  -> source of truth for WorldExplorer logic

Kaggle Dataset
  -> offline wheel mirror for Kaggle kernels

Slim kernel.py
  -> minimal bootstrapper
  -> installs from GitHub first when internet is enabled
  -> falls back to attached wheel/source when needed
  -> can be switched to wheel-first for offline notebooks
  -> falls back to attached source package if needed
```

The slim bootstrap now supports these acquisition policies:

```text
github_first  -> default; force-reinstall from GitHub, then attached wheel/source
wheel_first   -> offline path; attached wheel/source first, then GitHub if available
github_only   -> only try GitHub
wheel_only    -> only try attached wheel/source
```

Generated slim breaker kernels default to:

```text
repo = git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git@master
source_policy = github_first
enable_internet = true
```

For offline runs, the bootstrap searches attached Kaggle datasets for:

```text
wheelhouse/worldexplorer-*.whl
worldexplorer-*.whl
```

It installs the newest available wheel with:

```text
pip install --force-reinstall --no-deps <latest_worldexplorer_wheel>
```

## Why A Kaggle Dataset Still Exists

The Kaggle Dataset is not replacing GitHub.

It exists because the Kaggle kernel may not be allowed to reach GitHub at run
time. The dataset acts as the offline package cache. GitHub remains the primary
development and version-control location.

## Current State

Published GitHub repo:

```text
https://github.com/Amarel-Taylor-Scott/worldexplorer
```

Current commit:

```text
8191703 Bump WorldExplorer wheel version to 0.2.2
```

Current tag:

```text
v0.2.2
```

Published Kaggle wheel mirror:

```text
worldexplorer-0.2.2-py3-none-any.whl
```

## Bottom Line

I am not blocked from publishing to GitHub.

The thing that must be handled carefully is Kaggle execution mode:

```text
normal slim kernel:   GitHub-first
offline slim kernel:  wheel/source dataset-first
```

That keeps GitHub as the primary source of truth while still supporting
internet-disabled notebook requirements.
