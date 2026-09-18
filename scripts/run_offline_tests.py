"""Run the available test suite without Python network or dotenv access.

Usage: python -B scripts/run_offline_tests.py
Resolve the repository from this file, so invocation also works from another cwd.
These Python guards are not an OS sandbox for arbitrary native/subprocess code.
"""
import io
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
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(f'Interpreter: {sys.executable}\nPython: {sys.version.split()[0]}')
    print('Network and dotenv guards installed before test discovery.')
    print(stream.getvalue())
    print(f'Unexpected audited network/dotenv operations: {len(violations)}')
    sys.exit(0 if result.wasSuccessful() and result.testsRun and not violations else 1)
