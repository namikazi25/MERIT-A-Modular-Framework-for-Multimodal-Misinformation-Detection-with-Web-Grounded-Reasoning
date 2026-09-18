"""Shared, persistent admission for local discovery/extraction traffic.

These conservative ceilings are project policy, not guaranteed upstream quotas.
All redesign adapters use one state file across runs. Recorded access blocks and
interrupted leases require review; elapsed cooldown alone never clears them.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid

POLICY = {'search_interval_s': 30, 'extract_interval_s': 6,
          'target_interval_s': 30, 'max_in_flight_per_service': 1,
          'rate_cooldown_s': 3600, 'access_cooldown_s': 86400}
DEFAULT_PATH = Path(__file__).resolve().parents[2] / '.runtime/redesign-retrieval/traffic.json'


class TrafficBlocked(RuntimeError):
    def __init__(self, reason, retry_at=None):
        super().__init__(reason)
        self.reason = reason
        self.retry_at = retry_at


def retry_after_seconds(value, now):
    """Parse only Retry-After, without retaining arbitrary response headers."""
    if not isinstance(value, str):
        return 0
    try:
        seconds = float(value.strip())
        return max(0, seconds) if math.isfinite(seconds) else 0
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0, date.timestamp() - now)
        except (ValueError, TypeError, OverflowError):
            return 0


class Traffic:
    def __init__(self, path=DEFAULT_PATH, *, clock=time.time, sleep=time.sleep):
        self.path = Path(path)
        self.clock = clock
        self.sleep = sleep
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        existed = self.path.exists()
        with self.path.open('a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read()
                if raw:
                    try:
                        state = json.loads(raw)
                        if state['schema'] != 1 or state['policy'] != POLICY:
                            raise ValueError('Unsupported traffic policy')
                        if not isinstance(state['services'], dict) or not isinstance(state['scopes'], dict):
                            raise ValueError('Invalid traffic maps')
                        def valid_time(value):
                            return type(value) in (int, float) and math.isfinite(value) and value >= 0
                        if not valid_time(state['last_time']):
                            raise ValueError('Invalid traffic clock')
                        for entry in list(state['services'].values()) + list(state['scopes'].values()):
                            if not valid_time(entry['next_at']) or ('retry_at' in entry and not valid_time(entry['retry_at'])):
                                raise ValueError('Invalid traffic timestamp')
                        for entry in state['services'].values():
                            if entry['active'] is not None and (type(entry['active']) is not str or not entry['active']):
                                raise ValueError('Invalid traffic lease')
                    except (ValueError, KeyError, TypeError, OverflowError):
                        raise TrafficBlocked('Invalid traffic state or changed policy; review required')
                else:
                    if existed:
                        raise TrafficBlocked('Empty existing traffic state; review required')
                    state = {'schema': 1, 'policy': POLICY.copy(), 'services': {}, 'scopes': {}, 'last_time': 0}
                yield state
                f.seek(0)
                f.truncate()
                json.dump(state, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def start(self, service, scopes, *, deadline=None, max_wait=60):
        if service not in ('search', 'extract') or not scopes or any(type(s) is not str for s in scopes):
            raise ValueError('Explicit service and scopes required')
        started = self.clock()
        while True:
            with self.locked() as state:
                now = self.clock()
                if now < state['last_time']:
                    raise TrafficBlocked('Clock moved backwards; review required')
                state['last_time'] = now
                current = state['services'].setdefault(service, {'next_at': 0, 'active': None})
                if current['active']:
                    raise TrafficBlocked('Another request is active or interrupted; no parallel dispatch')
                entries = [state['scopes'].setdefault(s, {'next_at': 0}) for s in scopes]
                for entry in entries:
                    if entry.get('blocked'):
                        raise TrafficBlocked(entry['blocked'], entry.get('retry_at'))
                ready = max([current['next_at'], now] + [e['next_at'] for e in entries])
                if deadline and ready >= datetime.fromisoformat(deadline.replace('Z', '+00:00')).timestamp():
                    raise TrafficBlocked('Pacing would exceed the session deadline')
                delay = ready - now
                if delay > 0 and delay > max_wait - (now - started):
                    raise TrafficBlocked('Pacing wait exceeds bounded admission window', ready)
                if delay <= 0:
                    token = {'service': service, 'scopes': list(scopes), 'id': uuid.uuid4().hex}
                    current['active'] = token['id']
                    current['next_at'] = now + POLICY[service + '_interval_s']
                    for scope, entry in zip(scopes, entries):
                        interval = POLICY['target_interval_s'] if scope.startswith('host:') else POLICY[service + '_interval_s']
                        entry['next_at'] = now + interval
                    return token
            self.sleep(min(delay, 10))

    def finish(self, token, *, blocked_scopes=(), access=False, retry_after=None, reason=None):
        with self.locked() as state:
            current = state['services'][token['service']]
            if current['active'] != token['id']:
                raise TrafficBlocked('Request lease mismatch; review required')
            if any(s not in token['scopes'] for s in blocked_scopes):
                raise ValueError('Cannot block an unrelated scope')
            now = self.clock()
            delay = max(POLICY['access_cooldown_s'] if access else POLICY['rate_cooldown_s'],
                        retry_after_seconds(retry_after, now))
            for scope in blocked_scopes:
                entry = state['scopes'][scope]
                entry.update(blocked=reason or 'Access/rate failure requires review', retry_at=now + delay)
            current['active'] = None

    def record_observed_block(self, scope, observed_at, *, reason):
        """Import documented prior failures; this can only add protection."""
        with self.locked() as state:
            entry = state['scopes'].setdefault(scope, {'next_at': 0})
            entry.update(blocked=reason, retry_at=max(entry.get('retry_at', 0), observed_at + POLICY['access_cooldown_s']))
