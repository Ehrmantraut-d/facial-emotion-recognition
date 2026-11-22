"""Safely copy FER2013 CSV from a ZIP archive without extracting unneeded files."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "fer2013" / "fer2013.csv"


def import_archive(archive: Path) -> Path:
    archive = archive.expanduser().resolve(strict=True)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a ZIP archive: {archive}")
    if DESTINATION.exists():
        raise FileExistsError(f"Dataset already exists; refusing to overwrite: {DESTINATION}")
    with zipfile.ZipFile(archive) as zipped:
        matches = [item for item in zipped.infolist()
                   if Path(item.filename.replace("\\", "/")).name.lower() == "fer2013.csv"]
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError(f"Expected exactly one fer2013.csv member; found {len(matches)}")
        member = matches[0]
        if member.file_size < 10_000_000:
            raise ValueError("FER2013 CSV is unexpectedly small")
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        # Recover a complete interrupted extraction after validating its ZIP CRC.
        for candidate in DESTINATION.parent.glob("fer2013_*.part"):
            if candidate.stat().st_size != member.file_size:
                continue
            crc = 0
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    crc = zlib.crc32(chunk, crc)
            if (crc & 0xFFFFFFFF) == member.CRC:
                candidate.replace(DESTINATION)
                return DESTINATION
        with tempfile.NamedTemporaryFile(prefix="fer2013_", suffix=".part",
                                         dir=DESTINATION.parent, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with zipped.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if temporary.stat().st_size != member.file_size:
                raise IOError("Extracted CSV size does not match ZIP metadata")
            temporary.replace(DESTINATION)
        finally:
            temporary.unlink(missing_ok=True)
    return DESTINATION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Local ZIP containing FER2013 CSV")
    args = parser.parse_args()
    path = import_archive(args.archive)
    print(f"Imported FER2013 CSV ({path.stat().st_size:,} bytes) to {path}")


if __name__ == "__main__":
    main()
