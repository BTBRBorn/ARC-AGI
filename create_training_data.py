import argparse
from pathlib import Path
from get_tokenizer import Tokenizer
from get_dataloaders import create_data
from tqdm.auto import tqdm
import pickle
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="data/combined")
parser.add_argument("--tokenizer_data_folder", type=str, default="tokenizer_data/")
parser.add_argument("--num_shards", type=int, default=10)
parser.add_argument("--vocab_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument(
    "--tokenizer_save_path", type=str, default="tokenizers/tokenizer.pickle"
)

args = parser.parse_args()

if __name__ == "__main__":
    tokenizer = Tokenizer(args.vocab_size)

    TOKENIZER_DATA_FOLDER = Path(args.tokenizer_data_folder)
    DATA_PATH = Path(args.data_path)

    shutil.rmtree(DATA_PATH.parent / TOKENIZER_DATA_FOLDER)

    print("Data is being created.")
    for i in tqdm(range(1, args.num_shards + 1)):
        create_data(
            data_path=DATA_PATH,
            tokenizer=tokenizer,
            output_file_path=TOKENIZER_DATA_FOLDER / f"training_{i}.npy",
            is_train=True,
            rolled=True,
            augmented=True,
        )

    tokenizer.train(
        data_path=DATA_PATH.parent / TOKENIZER_DATA_FOLDER,
        num_workers=args.num_workers,
    )

    tokenizer_save_path = Path(args.tokenizer_save_path)
    if not tokenizer_save_path.parent.exists():
        tokenizer_save_path.parent.mkdir(parents=True)

    with open(tokenizer_save_path, "wb") as fhandle:
        pickle.dump(tokenizer, fhandle)
