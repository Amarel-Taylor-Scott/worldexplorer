# ----------------------------------------------------------------------------
# 1b. Metabolism (v12) -- the energy ledger that turns wall-clock into search
# ----------------------------------------------------------------------------

class Metabolism:
    """v12: the organism's energy ledger. ONE knob -- CFG.TIME_BUDGET_MIN --
    becomes the whole run plan:

      * a SHIPPING RESERVE is set aside first (forward gate, sealed audit,
        final refits, submission, reports): exploration can never eat it;
      * each spending phase opens a window sized as a SHARE OF WHAT REMAINS
        when it starts (marginal-value pacing -- a phase that finishes early
        donates its slack to every phase after it);
      * phase loops ask allow(phase) between units and stop on the clock;
      * the v10 circadian governor stays armed ~4% PAST the metabolic budget
        as the shed-cost backstop if shipping itself runs late.

    TIME_BUDGET_MIN = 0 disables everything: allow() is always True, no
    extra seasons/epochs run -- the fixed pacing of v11. Tiny budgets stay
    alive too: the reserve is capped at 35% of the budget so a 5-minute
    smoke still explores before it ships."""

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.enabled = float(cfg.TIME_BUDGET_MIN) > 0
        self.budget = float(cfg.TIME_BUDGET_MIN)
        self.reserve = (min(max(float(cfg.RESERVE_MIN), 0.06 * self.budget), 0.35 * self.budget)
                        if self.enabled else 0.0)
        self.usable_end = self.budget - self.reserve
        self.deadline: dict[str, float] = {}
        self.seasons = 0
        self.epochs = 0
        if self.enabled:
            log("metabolism_armed", time_budget_min=self.budget,
                shipping_reserve_min=round(self.reserve, 1),
                exploration_window_min=round(self.usable_end, 1))

    def now(self) -> float:
        return (time.monotonic() - RUN_START) / 60.0

    def begin(self, phase: str, share: float) -> float:
        """Open a spending window: deadline = now + share x (usable time left)."""
        if not self.enabled:
            return float("inf")
        t = self.now()
        d = min(self.usable_end, t + share * max(0.0, self.usable_end - t))
        self.deadline[phase] = d
        log("metabolism_window", phase=phase, now_min=round(t, 1),
            deadline_min=round(d, 1), share=share)
        self.heartbeat(f"begin_{phase}")
        return d

    def allow(self, phase: str) -> bool:
        """True while the phase may spend. Always True when disabled, so the
        disabled harness behaves exactly like v11's fixed budgets."""
        if not self.enabled:
            return True
        return self.now() < self.deadline.get(phase, self.usable_end)

    def wants_more(self, phase: str) -> bool:
        """Whether an EXTRA round (season/epoch/dive descent) should start.
        False when disabled -- extras only exist under a time budget."""
        return self.enabled and self.allow(phase)

    def heartbeat(self, tag: str) -> None:
        if not self.enabled:
            return
        write_json({"tag": tag, "elapsed_min": round(self.now(), 1),
                    "time_budget_min": self.budget,
                    "shipping_reserve_min": round(self.reserve, 1),
                    "seasons": self.seasons, "epochs": self.epochs,
                    "deadlines_min": {k: round(v, 1) for k, v in self.deadline.items()}},
                   "metabolism_heartbeat.json")


# Global metabolism instance (set once per run by the harness; None only
# before run() starts). Read by the evolution engine and the predator.
META: "Metabolism | None" = None


