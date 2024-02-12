import torch
import mlx.core as mx
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("filename", type=str, help="Filename to parse")
    args = args.parse_args()

    if ".pth" not in args.filename:
        raise ValueError(f"Expected a .pth file, got {args.filename}")
    filename = args.filename.split(".pth")[0]

    state_dict = torch.load(f"{filename}.pth", map_location=torch.device("cpu"))
    mx_state_dict = {}
    print("Converting pytorch format to SafeTensors")
    for k, v in state_dict.items():
        # numpy doesn't support bfloat16, convert to float32 first before converting to mlx
        mx_state_dict[k] = mx.array(v.to(torch.float32).numpy()).astype(mx.bfloat16)
    print("Saving to", f"{filename}.safetensors")
    mx.save_safetensors(f"{filename}.safetensors", mx_state_dict)
    