"""Prepare fixed-order log-Mel tensors for the source-group experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_split_manifest import read
from preprocess import wav_to_log_mel, normalize_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-root", type=Path, required=True,
                        help="Directory containing the field/train relative paths")
    parser.add_argument("--records", type=Path, default=Path(__file__).parent / "data/records.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read(args.records)
    features = np.stack([normalize_record(wav_to_log_mel(args.wav_root / r["relative_path"])) for r in rows])
    assert features.shape == (2485, 1, 48, 157) and np.isfinite(features).all()
    args.output.mkdir(parents=True)
    np.save(args.output / "features.npy", features)
    (args.output / "record_ids.json").write_text(json.dumps([r["record_id"] for r in rows]), encoding="utf8")


if __name__ == "__main__":
    main()
