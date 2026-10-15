"""Distinguish observable paused-prefill stalls from idle/unavailable samples."""

import math
import time


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class ProgressGuard:
    def __init__(self, timeout_s=300.0):
        self.timeout_s = float(timeout_s)
        self.signature = None
        self.last_advance = None
        self.last_observed = None

    def observe(self, sample, now, wall_now=None):
        if not isinstance(sample, dict):
            self.last_observed = None
            return None
        timestamp = sample.get("timestamp")
        wall = time.time() if wall_now is None else wall_now
        if (
            not finite_number(now)
            or not finite_number(wall)
            or not finite_number(timestamp)
            or not -5 <= wall - timestamp <= 15
        ):
            self.last_observed = None
            return None
        admission = sample.get("pd_admission")
        fair = admission.get("fair_prefill") if isinstance(admission, dict) else None
        if not isinstance(fair, dict):
            self.last_observed = None
            return None
        paused = fair.get("paused_contexts")
        if not finite_number(paused) or paused < 0 or int(paused) != paused:
            self.last_observed = None
            return None
        completed = fair.get("last_completed_ts")
        busy = sample.get("total_prefill_busy_us")
        if (
            completed is not None and (not finite_number(completed) or completed < 0)
        ) or (not finite_number(busy) or busy < 0):
            self.last_observed = None
            return None
        if paused <= 0:
            self.signature = None
            self.last_advance = None
            self.last_observed = now
            return False
        signature = (completed, busy)
        discontinuous = self.last_observed is None or now - self.last_observed > 30.0
        self.last_observed = now
        if signature != self.signature or self.last_advance is None or discontinuous:
            self.signature = signature
            self.last_advance = now
            return False
        return now - self.last_advance >= self.timeout_s
