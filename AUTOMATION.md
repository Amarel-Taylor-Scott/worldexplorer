# worldexplorer — automation, publishing & submitting

Everything external (publish code, run on Kaggle, submit) goes through the
repo tools. `tools/publish.py` reads credentials from the local vault
(`~/.config/worldexplorer/.env`), while normal Git pushes use the configured Git
remote/credential helper.

GitHub is the source of truth for all WorldExplorer logic. Kaggle kernels should
be slim launchers that fetch this repo, then call `wx.kaggle.run(CONFIG)`.

---

## A. Publish source to GitHub

**Publish the repo to GitHub** (creates the repo if needed, pushes, leaves a
clean token-less remote):

```bash
cd /home/username/new_algo/worldexplorer && python tools/publish.py github --repo worldexplorer
```

Or push ordinary commits:

```bash
cd /home/username/new_algo/worldexplorer && git push origin master
```

## B. Publish the offline wheel/source mirror

Use this only for Internet-OFF notebooks where pip-from-GitHub cannot run:

```bash
cd /home/username/new_algo/worldexplorer && python tools/publish.py kaggle-dataset --slug worldexplorer-engine
```

## C. Push and run slim Kaggle kernels

GitHub-first, internet-enabled kernel:

```bash
cd /home/username/new_algo/worldexplorer
python tools/fleet.py bootstrap --name wx-github-master --repo-ref master --time-budget 120
python tools/fleet.py push --manifest /home/username/new_algo/kaggle/fleet/wx-github-master_manifest.json
```

Offline, wheel/source-backed kernel:

```bash
cd /home/username/new_algo/worldexplorer
python tools/fleet.py bootstrap \
  --name wx-offline-wheel \
  --offline \
  --source-policy wheel_first \
  --engine-dataset /kaggle/input/worldexplorer-engine \
  --dataset taylorsamarel/worldexplorer-engine \
  --repo-ref master \
  --time-budget 120
python tools/fleet.py push --manifest /home/username/new_algo/kaggle/fleet/wx-offline-wheel_manifest.json
```

## D. Submit a predictions CSV

```bash
python tools/publish.py submit /home/username/drw_out/submission.csv -m "v36 run"
```

---

## E. The thin Kaggle kernel

- **Internet ON** competitions: generate a slim kernel instead of pasting the
  engine:
  ```bash
  python tools/fleet.py bootstrap --name wx-github-master \
    --repo git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git \
    --repo-ref master --time-budget 120
  python tools/fleet.py push --manifest /home/username/new_algo/kaggle/fleet/wx-github-master_manifest.json
  ```
- **Internet OFF** (DRW, code competitions): `Add Input` → your
  `worldexplorer-engine` dataset, generate with `--offline --source-policy
  wheel_first`, then paste or push `kaggle/bootstrap_kernel.py` through the
  fleet tooling. See `PUBLISH.md` for the full flow.

---

## F. Credentials

The vault is `~/.config/worldexplorer/.env` (chmod 600), git-ignored, holding
`GITHUB_OWNER/GITHUB_TOKEN` and `KAGGLE_USERNAME/KAGGLE_API_TOKEN`. `publish.py`
and the `kaggle` CLI read it automatically; `git push` uses the stored git
credential. Nothing here is ever committed.
