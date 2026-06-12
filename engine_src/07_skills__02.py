def _predict_codebook(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    code = state["km"].predict((Z - state["mu0"]) / state["sd0"])
    return state["table"][code].astype(np.float32)


def _predict_router(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    base = np.asarray(state["model"].predict(Z), np.float64)
    if state["experts"] and X_raw is not None and ATLAS is not None:
        t_ids = ATLAS.assign(X_raw)
        for c, (e, n_c) in state["experts"].items():
            m = t_ids == c
            if m.any():
                wgt = n_c / (n_c + state["shrink_n"])
                base[m] = (1.0 - wgt) * base[m] + wgt * np.asarray(e.predict(Z[m]), np.float64)
    return base.astype(np.float32)


def _predict_steep(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    d = np.asarray(state["model"].predict(Z), np.float64)
    s = (np.asarray(state["steep"].predict(Z), np.float64) - state["smu"]) / state["ssd"]
    return (d * (1.0 + 0.5 * np.clip(s, -1.5, 1.5))).astype(np.float32)


def _predict_scout(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    if state["scouts"]:
        tot = sum(w0 for _, _, w0 in state["scouts"])
        out = np.zeros(len(Z), np.float64)
        for sub, m, w0 in state["scouts"]:
            p = np.asarray(m.predict(Z[:, sub]), np.float64)
            out += (w0 / tot) * (p - p.mean()) / (p.std() + 1e-9)
        return out.astype(np.float32)
    return np.asarray(state["model"].predict(Z), np.float32)


def _predict_beta_linear(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    # shared by 'relay' and 'ladder': standardized Z through a fitted beta
    Zs = (Z - state["mu"]) / state["sd"]
    return (Zs.astype(np.float64) @ state["beta"].astype(np.float64)).astype(np.float32)


def _predict_greedyols(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    return np.asarray(state["model"].predict(Z[:, state["sel"]]), np.float32)


def _predict_linpearson(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    if state.get("torch_net"):
        Zsc = state["scaler"].transform(Z)
        return _predict_torch_mlp(state["model"], np.asarray(Zsc, np.float32))
    return np.asarray(state["model"].predict(Z), np.float32)


def _predict_gpuswarm(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    return _predict_gpu_swarm(state, Z)


def _predict_reservoir(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    Zs = ((Z - state["res_mu"]) / state["res_sd"]).astype(np.float32)
    res, a, W_in, W = state["res_size"], state["leak"], state["W_in"], state["W"]
    h = np.zeros(res, np.float32)
    H = np.empty((len(Zs), res), np.float32)
    drive = Zs @ W_in
    for t in range(len(Zs)):
        h = (1.0 - a) * h + a * np.tanh(drive[t] + h @ W)
        H[t] = h
    return np.asarray(state["model"].predict(H), np.float32)


def _predict_terrace(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    s = np.asarray(state["model"].predict(Z), np.float64)
    steps = state["lut"][np.clip(np.searchsorted(state["edges"], s, side="right"),
                                 0, len(state["lut"]) - 1)].astype(np.float64)
    return (0.5 * (s - state["smu"]) / state["ssd"]
            + 0.5 * (steps - state["lmu"]) / state["lsd"]).astype(np.float32)


def _predict_rapids(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    p = np.asarray(state["model"].predict(Z), np.float64)
    if state["m2"] is not None and state["w2"] > 0:
        p = p + state["w2"] * np.asarray(state["m2"].predict(Z), np.float64)
    return p.astype(np.float32)


def _predict_slope_single(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    # shared by 'uni' and 'tsen': one feature, one slope, one center
    return (state["slope"] * (Z[:, state["j"]].astype(np.float64) - state["mu"])).astype(np.float32)


def _predict_binmean(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    codes = np.searchsorted(state["edges"], Z[:, state["j"]], side="right")
    return state["means"][np.clip(codes, 0, len(state["means"]) - 1)].astype(np.float32)


def _predict_vote(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    votes = np.sign(Z - state["med"]) * state["signs"]
    return votes.mean(axis=1).astype(np.float32)


def _predict_knn(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    return state["model"].predict(state["scaler"].transform(Z[:, : state["d"]])).astype(np.float32)


def _predict_ridgebag(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    return np.mean([m.predict(Z) for m in state["models"]], axis=0).astype(np.float32)


def _predict_mlp(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    Zs = state["scaler"].transform(Z)
    if state.get("torch_mlp"):
        return _predict_torch_mlp(state["model"], np.asarray(Zs, np.float32))
    return state["model"].predict(Zs).astype(np.float32)


def _predict_pls(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    # v27: PLS predict is 2-D -> flatten
    return np.asarray(state["model"].predict(Z), np.float32).ravel()


def _predict_default(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    # every plain state["model"] skill: ridge/ols/huber/elastic/bayesridge/ard/
    # recw/shiftlin/recency/hgb/gbdt/swell/...
    return np.asarray(state["model"].predict(Z), np.float32)


# Predict-side dispatch. Kinds NOT listed here fall through to
# _predict_default (plain state["model"].predict) -- the same fall-through the
# old if-chain had.
_PREDICT: dict[str, Callable[..., np.ndarray]] = {
    "codebook": _predict_codebook,
    "router": _predict_router,
    "steep": _predict_steep,
    "scout": _predict_scout,
    "relay": _predict_beta_linear,
    "ladder": _predict_beta_linear,
    "greedyols": _predict_greedyols,
    "linpearson": _predict_linpearson,
    "gpuswarm": _predict_gpuswarm,
    "reservoir": _predict_reservoir,
    "terrace": _predict_terrace,
    "rapids": _predict_rapids,
    "uni": _predict_slope_single,
    "tsen": _predict_slope_single,
    "binmean": _predict_binmean,
    "vote": _predict_vote,
    "knn": _predict_knn,
    "ridgebag": _predict_ridgebag,
    "mlp": _predict_mlp,
    "pls": _predict_pls,
}


def _predict_core(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None = None) -> np.ndarray:
    return _PREDICT.get(state["kind"], _predict_default)(state, Z, X_raw)


def predict_skill(state: dict[str, Any], X_any: np.ndarray) -> np.ndarray:
    Z = state["tf"](X_any[:, state["idx"]])
    return _predict_core(state, Z, X_any)


def stability_probe(state: dict[str, Any], X_va: np.ndarray, base_pred: np.ndarray,
                    rng: np.random.Generator, cfg: HarnessConfig) -> float:
    Z = state["tf"](X_va[:, state["idx"]]).copy()
    col_sd = Z.std(axis=0, keepdims=True) + 1e-6
    Zn = Z + rng.normal(0, cfg.STABILITY_NOISE, Z.shape).astype(np.float32) * col_sd
    p = _predict_core(state, Zn, X_va)    # terrain ids stay honest (unperturbed X)
    scale = float(np.std(base_pred)) + 1e-9
    return float(np.sqrt(np.mean((p - base_pred) ** 2)) / scale)


