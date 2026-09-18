"""Append-only, process-safe reservations. Ambiguous requests remain reserved on resume."""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import fcntl
import hashlib
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
            events = self.events(f)
            existing = self.effective_policy(events)
            if existing and existing != policy:
                raise BudgetBlocked('Ledger policy differs on resume')
            if not existing:
                self.append(f, policy)

    @staticmethod
    def effective_policy(events):
        """A renewal changes only the deadline, with a rechecked approval record."""
        policy = None
        for event in events:
            if event['event'] == 'ledger_policy':
                if policy is not None:
                    raise BudgetBlocked('Duplicate ledger policy')
                policy = dict(event)
            elif event['event'] == 'deadline_renewal':
                if policy is None or event['previous_deadline'] != policy['deadline']:
                    raise BudgetBlocked('Invalid deadline renewal chain')
                path = Path(event['authorization_path'])
                if path.name.startswith('.env'):
                    raise BudgetBlocked('Environment files are not authorization records')
                try:
                    data = path.read_bytes()
                    record = json.loads(data)
                    start = datetime.fromisoformat(record['window_start_utc'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(record['deadline_utc'].replace('Z', '+00:00'))
                    old = datetime.fromisoformat(record['previous_deadline_utc'].replace('Z', '+00:00'))
                    valid = (hashlib.sha256(data).hexdigest() == event['authorization_sha256']
                             and start.tzinfo is not None and end.tzinfo is not None
                             and 0 < (end - start).total_seconds() <= 7200 and end > old
                             and record['previous_deadline_utc'] == policy['deadline']
                             and record['deadline_utc'] == event['deadline']
                             and money(record['combined_cap_usd']) == money(policy['cap'])
                             and record['new_budget_grant'] is False
                             and type(record['authorization']) is str and bool(record['authorization'].strip())
                             and type(record['user_reply']) is str and bool(record['user_reply'].strip()))
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
                if not valid:
                    raise BudgetBlocked('Missing or changed deadline authorization record')
                policy['deadline'] = event['deadline']
        return policy

    def renew_deadline(self, authorization_path):
        """Operator-only action after explicit user approval; never called by a client.

        Approval provenance is an audit reference, not protection against an actor
        who can rewrite the ledger and all approval records. Budgets never reset.
        """
        path = Path(authorization_path).resolve()
        if path.name.startswith('.env'):
            raise BudgetBlocked('Environment files are not authorization records')
        data = path.read_bytes()
        record = json.loads(data)
        with self.locked() as f:
            events = self.events(f)
            policy = self.effective_policy(events)
            _, pending, halted = self.balance(events)
            if pending or halted or policy['deadline'] != self.deadline:
                raise BudgetBlocked('Reconcile pending, halted or stale ledger before renewal')
            event = {'event': 'deadline_renewal', 'previous_deadline': self.deadline,
                     'deadline': record['deadline_utc'], 'authorization_path': str(path),
                     'authorization_sha256': hashlib.sha256(data).hexdigest()}
            updated = self.effective_policy(events + [event])
            if datetime.now(timezone.utc) >= datetime.fromisoformat(updated['deadline'].replace('Z', '+00:00')):
                raise BudgetBlocked('Renewed window already expired')
            self.append(f, event)
            self.deadline = updated['deadline']

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
            events = self.events(f)
            policy = self.effective_policy(events)
            if policy['deadline'] != self.deadline:
                raise BudgetBlocked('Stale deadline; reopen the verified ledger')
            spent, pending, halted = self.balance(events)
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
            events = self.events(f)
            policy = self.effective_policy(events)
            if policy['deadline'] != self.deadline:
                raise BudgetBlocked('Stale deadline; reopen the verified ledger')
            spent, pending, halted = self.balance(events)
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
