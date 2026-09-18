"""Run the available test suite without Python network or dotenv access.

Usage: python -B scripts/run_offline_tests.py
Resolve the repository from this file, so invocation also works from another cwd.
These Python guards are not an OS sandbox for arbitrary native/subprocess code.
"""
import io
import argparse
import os
from pathlib import Path
import socket
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.chdir(ROOT)
for name in list(os.environ):
    if any(part in name.upper() for part in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
        os.environ.pop(name, None)

violations = []


def audit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        violations.append(event)
        raise AssertionError('Offline test blocked a network operation')
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).name.startswith('.env'):
            violations.append('dotenv file access')
            raise AssertionError('Offline test blocked an environment-file read')


sys.addaudithook(audit)


def no_network(*args, **kwargs):
    violations.append('network operation')
    raise AssertionError('Offline test blocked a network operation')


socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network
dotenv = types.ModuleType('dotenv')
dotenv.load_dotenv = lambda *args, **kwargs: False
sys.modules['dotenv'] = dotenv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('full', 'portable'), default='full',
                        help='full includes private artifact checks; portable runs synthetic source-only tests')
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    excluded = []
    if args.profile == 'portable':
        private_classes = {
            'test_preflight_corrections.HistoricalPreservationTest',
            'test_preflight_corrections.PreFixContrastTest',
        }
        def select(tests):
            for test in tests:
                if isinstance(test, unittest.TestSuite):
                    yield from select(test)
                elif test.id().rsplit('.', 1)[0] in private_classes:
                    excluded.append(test.id())
                else:
                    yield test
        suite = unittest.TestSuite(select(suite))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(f'Interpreter: {sys.executable}\nPython: {sys.version.split()[0]}')
    print(f'Profile: {args.profile}; explicitly excluded private-artifact checks: {len(excluded)}')
    for ident in excluded:
        print(f'EXCLUDED (requires preserved local artifacts): {ident}')
    print('Network and dotenv guards installed before test discovery.')
    print(stream.getvalue())
    print(f'Unexpected audited network/dotenv operations: {len(violations)}')
    sys.exit(0 if result.wasSuccessful() and result.testsRun and not violations else 1)
