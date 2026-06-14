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
3. An internet-disabled Kaggle kernel cannot reliably `pip install` directly
   from GitHub during execution.
4. Therefore the slim Kaggle kernel should load the latest attached wheel from
   a Kaggle Dataset mirror first, then fall back to GitHub only when internet is
   available.

## Current Implemented Setup

The intended structure is:

```text
GitHub repo
  -> source of truth for WorldExplorer logic

Kaggle Dataset
  -> offline wheel mirror for Kaggle kernels

Slim kernel.py
  -> minimal bootstrapper
  -> installs latest attached wheel first
  -> falls back to GitHub install when internet is available
  -> falls back to attached source package if needed
```

The slim bootstrap now searches attached Kaggle datasets for:

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

The thing that must be handled carefully is Kaggle execution: for offline
Kaggle kernels, the slim kernel has to install from the latest attached wheel,
not assume it can pull live code from GitHub.
