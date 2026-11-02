"""Download the public credit card fraud dataset (about 150 MB) to data/creditcard.csv."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fraud.data import DATA_URL, DEFAULT_PATH  # noqa: E402


def main() -> None:
    if DEFAULT_PATH.exists():
        print(f"Already downloaded: {DEFAULT_PATH}")
        return
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, DEFAULT_PATH)
    print(f"Saved to {DEFAULT_PATH} ({DEFAULT_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
