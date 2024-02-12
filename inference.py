from rwkv import RWKV
from tokenizer import TrieTokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Folder with the model files")
    args = parser.parse_args()

    model = RWKV.load_weights(args.model)
    tokenizer = TrieTokenizer(f"{args.model}/rwkv_vocab_v20230424.txt")