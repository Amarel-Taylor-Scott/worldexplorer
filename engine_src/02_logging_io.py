# ----------------------------------------------------------------------------
# 1. Logging / IO
# ----------------------------------------------------------------------------

def log(event: str, **kw: Any) -> None:
    parts = " | ".join(f"{k}={v}" for k, v in kw.items())
    print(f"{time.monotonic() - RUN_START:8.1f}s {event}" + (f" | {parts}" if parts else ""), flush=True)


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)


def write_json(obj: Any, name: str) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:8], 16)


