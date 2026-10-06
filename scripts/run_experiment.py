"""Direct Windows/Linux entry point, also usable without editable installation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rotnet.experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
