# ----------------------------------------------------------------------------
# 3. Data: discovery, synthetic fixture, features, matrices
# ----------------------------------------------------------------------------

def find_data_root() -> Path | None:
    for root in CFG.DATA_ROOTS:
        p = Path(root)
        if (p / "train.parquet").exists() and (p / "test.parquet").exists():
            return p
    return None


def make_synthetic(rows: int, n_anon: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def block_world(n: int, shift: bool) -> pd.DataFrame:
        n_factors = 6
        F = rng.normal(size=(n, n_factors)).astype(np.float32)
        cols = {}
        per_block = max(2, n_anon // n_factors)
        k = 0
        for f in range(n_factors):
            for _ in range(per_block):
                if k >= n_anon:
                    break
                cols[f"X{k + 1}"] = F[:, f] * rng.uniform(0.7, 1.3) + rng.normal(0, 0.45, n).astype(np.float32)
                k += 1
        while k < n_anon:
            cols[f"X{k + 1}"] = rng.normal(0, 1, n).astype(np.float32)
            k += 1
        w = np.array([0.10, -0.07, 0.05, 0.0, 0.0, 0.0])
        if shift:
            w = np.array([0.02, -0.02, 0.0, 0.09, -0.06, 0.0])
        label = (F @ w + rng.normal(0, 1.0, n)).astype(np.float32)
        df = pd.DataFrame(cols)
        df["bid_qty"] = np.abs(rng.lognormal(1.5, 0.8, n)).astype(np.float32)
        df["ask_qty"] = np.abs(rng.lognormal(1.5, 0.8, n)).astype(np.float32)
        df["buy_qty"] = np.abs(rng.lognormal(1.0, 1.0, n)).astype(np.float32)
        df["sell_qty"] = np.abs(rng.lognormal(1.0, 1.0, n)).astype(np.float32)
        df["volume"] = (df["buy_qty"] + df["sell_qty"] + np.abs(rng.lognormal(1.2, 0.7, n))).astype(np.float32)
        df["label"] = label
        return df

    half = rows // 2
    train = pd.concat([block_world(half, False), block_world(rows - half, True)], ignore_index=True)
    test = block_world(max(2000, rows // 4), True).drop(columns=["label"])
    return train, test


def load_competition(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("loading_train_parquet")
    train = pd.read_parquet(root / "train.parquet").reset_index(drop=True)
    log("loading_test_parquet")
    test = pd.read_parquet(root / "test.parquet").reset_index(drop=True)
    if "label" in test.columns:
        test = test.drop(columns=["label"])
    for df, name in ((train, "train"), (test, "test")):
        f64 = [c for c in df.columns if df[c].dtype == np.float64]
        for c in f64:
            df[c] = df[c].astype(np.float32)
        log("memory_reduced", frame=name, rows=len(df), cols=df.shape[1])
    return train, test


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    need = {"bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"}
    if not need.issubset(df.columns):
        return pd.DataFrame(index=df.index)
    eps = np.float32(1e-8)
    b, a = df["bid_qty"].to_numpy(np.float32), df["ask_qty"].to_numpy(np.float32)
    bu, se = df["buy_qty"].to_numpy(np.float32), df["sell_qty"].to_numpy(np.float32)
    v = df["volume"].to_numpy(np.float32)
    depth, trades, flow = b + a, bu + se, bu - se
    feats = {
        "mkt_spread_proxy": a - b, "mkt_total_depth": depth, "mkt_net_flow": flow,
        "mkt_total_trades": trades, "mkt_vol_per_trade": v / (trades + eps),
        "mkt_buy_ratio": bu / (trades + eps), "mkt_order_imbalance": (b - a) / (depth + eps),
        "mkt_flow_imbalance": flow / (trades + eps), "mkt_kyle_lambda": np.abs(flow) / (v + eps),
        "mkt_depth_depletion": v / (depth + eps), "mkt_fill_probability": v / (trades + eps),
        "mkt_log_volume": np.log1p(v), "mkt_log_depth": np.log1p(depth),
        "mkt_stress": v / (depth + eps) * np.abs(flow) / (trades + eps),
        "mkt_signed_volume": np.sign(flow) * v,
    }
    out = pd.DataFrame(feats, index=df.index).astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_matrix(df: pd.DataFrame, cols: list[str], medians: pd.Series) -> np.ndarray:
    x = df.reindex(columns=cols).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(medians).fillna(0.0)
    return np.ascontiguousarray(x.to_numpy(np.float32))


