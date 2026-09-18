"""Compare installed test dependencies with the recorded local version snapshot.

No installation, network, environment-file access or package URL inspection.
This checks names/versions, not wheel hashes or cross-platform compatibility.
"""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform


def compare(reference):
    differences = []
    for name, expected in reference['packages'].items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = None
        if actual != expected:
            differences.append({'package': name, 'expected': expected, 'actual': actual})
    current = {'python': platform.python_version(), 'implementation': platform.python_implementation(),
               'system': platform.system(), 'machine': platform.machine()}
    platform_matches = current == reference['platform']
    return {'status': 'MATCH' if not differences and platform_matches else 'DIFFERS',
            'checked_packages': len(reference['packages']), 'package_differences': differences,
            'platform_matches': platform_matches, 'current_platform': current,
            'limitation': 'Installed-version comparison only; no wheel integrity, installation or live service validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=str(Path(__file__).resolve().parents[1] / 'config/offline_test_environment.json'))
    args = parser.parse_args()
    result = compare(json.loads(Path(args.reference).read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'MATCH' else 1)


if __name__ == '__main__':
    main()
