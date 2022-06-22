import itertools
from pathlib import Path

def gather_reqs(package_path):
    package_path = Path(package_path)

    reqs = (package_path / 'requirements.txt').read_text().splitlines()

    extras = {}

    for subpackage in package_path.iterdir():
        try:
            subrequirements = (subpackage / 'requirements.txt').read_text()
            extras[subpackage.name] = subrequirements.splitlines()
        except (FileNotFoundError, NotADirectoryError):
            pass

    extras['all'] = list(
        itertools.chain(reqs for subpackage, reqs in extras.items())
    )

    return reqs, extras