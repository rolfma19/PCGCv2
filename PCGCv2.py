import os
import re
import argparse
import glob


def run_pcgcv2(ply_path):
    python_exec = "/home/whma/miniconda3/envs/pcgcv2/bin/python"
    ckpt_path="ckpts/r3_0.10bpp.pth"

    cmd = f"{python_exec} coder.py --filedir={ply_path} --ckptdir={ckpt_path} --scaling_factor=1.0 --rho=1.0 --res=1024"

    result = os.popen(cmd)
    output = result.read()
    return output


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--ply-dir", type=str, default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ply_dir = ""
    args = parse_args()
    if args.ply_dir != "":
        ply_dir = args.ply_dir
    if os.path.isdir(ply_dir):
        ply_list = sorted(
            glob.glob(os.path.join(ply_dir, "**/" + "*.ply"), recursive=True)
        )

        total_bpp = 0
        for ply_path in ply_list:
            output = run_pcgcv2(ply_path)
            res = re.search(r"bpps:.*\nWrite", output)
            res = re.search("[-+]?[0-9]*\.?[0-9]+",res.group(0))        
            bpp = float(res.group(0).split(" ")[0])
            total_bpp += bpp
            print(f"{ply_path} : bpp {bpp}")
        print(f"avg bpp: {total_bpp/len(ply_list)}")
    else:
        ply_path = ply_dir
        output = run_pcgcv2(ply_path)
        res = re.search(r"bpps:.*\nWrite", output)
        res = re.search("[-+]?[0-9]*\.?[0-9]+",res.group(0))        
        bpp = float(res.group(0).split(" ")[0])
        print(f"{ply_path} : bpp {bpp}")
