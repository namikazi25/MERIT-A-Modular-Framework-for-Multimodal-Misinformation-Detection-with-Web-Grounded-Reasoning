"""Append-only, process-safe reservations. Ambiguous requests remain reserved on resume."""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import fcntl
import json
import os
import uuid


class BudgetBlocked(RuntimeError):
    pass


def money(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError('Cost must be finite and nonnegative')
    return amount


class Ledger:
    def __init__(self, path, cap, *, applicable_remaining=None, deadline=None, concurrency=2):
        self.path = Path(path)
        self.cap = money(cap)
        self.remaining = None if applicable_remaining is None else money(applicable_remaining)
        self.deadline = deadline
        if type(concurrency) is not int or not 1 <= concurrency <= 2:
            raise ValueError('Concurrency must be one or two')
        self.concurrency = concurrency
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Persist policy once; a resume must not reset budget or extend the deadline.
        policy = {'event': 'ledger_policy', 'cap': str(self.cap),
                  'applicable_remaining': None if self.remaining is None else str(self.remaining),
                  'deadline': deadline, 'concurrency': concurrency}
        with self.locked() as f:
            policies = [e for e in self.events(f) if e['event'] == 'ledger_policy']
            if policies and any(e != policy for e in policies):
                raise BudgetBlocked('Ledger policy differs on resume')
            if not policies:
                self.append(f, policy)

    @contextmanager
    def locked(self):
        with self.path.open('a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def events(f):
        f.seek(0)
        return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def append(f, event):
        f.seek(0, 2)
        f.write(json.dumps(event, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())

    @staticmethod
    def balance(events):
        pending, spent = {}, Decimal(0)
        halted = False
        for e in events:
            if e['event'] == 'reserve':
                pending[e['request_id']] = money(e['maximum_usd'])
            elif e['event'] == 'settle':
                pending.pop(e['request_id'])
                spent += money(e['actual_usd'])
            elif e['event'] == 'halt':
                halted = True
        return spent, pending, halted

    def summary(self):
        with self.locked() as f:
            spent, pending, halted = self.balance(self.events(f))
        effective = self.cap if self.remaining is None else min(self.cap, self.remaining)
        return {'spent_usd': str(spent), 'reserved_usd': str(sum(pending.values(), Decimal(0))),
                'pending_ids': list(pending), 'halted': halted,
                'session_available_usd': str(effective - spent - sum(pending.values(), Decimal(0))),
                'paid_dispatch_ready': self.remaining is not None and not halted}

    def reserve(self, service, maximum_usd, *, retries=0):
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError('At most two retries are permitted')
        bound = money(maximum_usd) * (retries + 1)
        with self.locked() as f:
            spent, pending, halted = self.balance(self.events(f))
            if halted:
                raise BudgetBlocked('Ledger halted after billing failure')
            if self.deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(self.deadline.replace('Z', '+00:00')):
                raise BudgetBlocked('Session deadline reached')
            if bound > 0 and self.remaining is None:
                raise BudgetBlocked('Applicable pre-existing remaining budget is unreconciled')
            if len(pending) >= self.concurrency:
                raise BudgetBlocked('In-flight concurrency limit reached')
            cap = self.cap if self.remaining is None else min(self.cap, self.remaining)
            if spent + sum(pending.values(), Decimal(0)) + bound > cap:
                raise BudgetBlocked('Insufficient budget including in-flight reservations')
            ident = uuid.uuid4().hex
            self.append(f, {'event': 'reserve', 'request_id': ident, 'service': service,
                            'maximum_usd': str(bound), 'retries': retries,
                            'time': datetime.now(timezone.utc).isoformat()})
            return ident

    def settle(self, request_id, actual_usd, usage=None):
        actual = money(actual_usd)
        with self.locked() as f:
            _, pending, _ = self.balance(self.events(f))
            if request_id not in pending:
                raise BudgetBlocked('Unknown or already settled request')
            if actual > pending[request_id]:
                self.append(f, {'event': 'halt', 'reason': 'actual_exceeded_reservation',
                                'request_id': request_id, 'reported_usd': str(actual)})
                raise BudgetBlocked('Actual exceeded reservation; manual reconciliation required')
            self.append(f, {'event': 'settle', 'request_id': request_id,
                            'actual_usd': str(actual), 'usage': usage or {}})

    def halt(self, reason):
        with self.locked() as f:
            self.append(f, {'event': 'halt', 'reason': reason})
