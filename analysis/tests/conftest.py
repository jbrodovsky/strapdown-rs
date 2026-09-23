"""Shared test configuration for the ``analysis`` package.

Running pytest from the repository root puts that root on ``sys.path``, where the
``analysis/`` *directory* shadows the real package at ``analysis/src/analysis`` as an implicit
namespace package -- ``import analysis`` then resolves to an empty namespace with none of the
package's contents. Putting the source root first makes the import resolve to the package
under test no matter which directory pytest was invoked from.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

if sys.path[:1] != [str(SRC)]:
    sys.path.insert(0, str(SRC))
