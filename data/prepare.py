import json
from pathlib import Path
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str)
    parser.add_argument("--target_dir", type=str)

    args = parser.parse_args()

    source_file, target_dir = Path(args.source_file), Path(args.target_dir)

    if target_dir.exists():
        shutil.rmtree(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
    else:
        target_dir.mkdir(exist_ok=True, parents=True)

    with open(source_file, "r") as fhandle:
        source_json = json.load(fhandle)

    for key in source_json.keys():
        file_name = key + ".json"
        file_path = target_dir / file_name
        with open(file_path, "w") as fhandle:
            json.dump(source_json[key], fhandle)
