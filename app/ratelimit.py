"""Per-IP daily request limit for the endpoints that cost Gemini quota.

Deliberately small: a counter per IP in process memory, reset when the UTC day changes.
Consequences worth knowing:
  * It is per process. A restart (Render free tier restarts and sleeps) clears the counters, and
    several workers would each count on their own. One uvicorn process is what this app runs.
  * It limits requests, not tokens: a cached summary or quiz still counts as one request.
  * IP is a weak identity (shared NAT, IPv6 rotation). It protects a free-tier quota from casual
    abuse; it is not authentication.
"""

import threading
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

# Keys are client addresses, so the table must not grow without bound. When it is full, addresses we
# have not seen yet are refused (fail closed: this guards a paid-for quota). It resets every day.
MAX_TRACKED_IPS = 50_000


@dataclass(frozen=True, slots=True)
class RateDecision:
    allowed: bool
    limit: int
    used: int  # requests counted for this IP today, including this one if it was allowed
    reset_in_seconds: int  # until the next UTC midnight


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DailyRateLimiter:
    """At most `limit` requests per key per UTC day. Thread-safe (endpoints run in a thread pool)."""

    def __init__(self, limit: int, *, clock: Callable[[], datetime] = _utc_now) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = limit
        self._clock = clock
        self._lock = threading.Lock()
        self._day = clock().date()
        self._counts: dict[str, int] = {}

    def hit(self, key: str) -> RateDecision:
        """Count one request for `key` and say whether it may proceed. Refused requests are not counted."""
        now = self._clock()
        with self._lock:
            if now.date() != self._day:
                self._day = now.date()
                self._counts.clear()
            used = self._counts.get(key, 0)
            reset = self._seconds_until_midnight(now)
            if used >= self.limit or (key not in self._counts and len(self._counts) >= MAX_TRACKED_IPS):
                return RateDecision(False, self.limit, used, reset)
            self._counts[key] = used + 1
            return RateDecision(True, self.limit, used + 1, reset)

    @staticmethod
    def _seconds_until_midnight(now: datetime) -> int:
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        return max(1, int((midnight - now).total_seconds()))


def client_ip(scope: MutableMapping[str, Any], trusted_hops: int) -> str:
    """The address the daily limit is counted against.

    trusted_hops = 0: the socket peer. Otherwise the entry `trusted_hops` places from the right of
    X-Forwarded-For: each trusted proxy appends the address it received the request from, so the
    right-hand entries are written by our own infrastructure, while anything further left may have
    been sent by the client. If the header is shorter than expected it is not trusted at all.
    """
    client = scope.get("client")
    peer = client[0] if client else "unknown"
    if trusted_hops <= 0:
        return peer
    headers: Mapping[bytes, bytes] = dict(scope.get("headers", []))
    forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1")
    parts = [part.strip() for part in forwarded.split(",") if part.strip()]
    if len(parts) >= trusted_hops:
        return parts[-trusted_hops]
    return peer
