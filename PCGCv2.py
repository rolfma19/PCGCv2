import os
import re
import argparse
import glob
from concurrent.futures.process import ProcessPoolExecutor

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

def run_pcgcv2_one(ply_path):
    output=run_pcgcv2(ply_path)
    res = re.search(r"bpps:.*\nWrite", output)
    res = re.search("[-+]?[0-9]*\.?[0-9]+",res.group(0))        
    bpp = float(res.group(0).split(" ")[0])
    return bpp

if __name__ == "__main__":
    ply_dir = ""
    args = parse_args()
    if args.ply_dir != "":
        ply_dir = args.ply_dir
    if os.path.isdir(ply_dir):
        ply_list = sorted(
            glob.glob(os.path.join(ply_dir, "**/" + "*.ply"), recursive=True)
        )
        pool=ProcessPoolExecutor(16)
        total_bpp = 0
        for ply_path in ply_list:           
            bpp = pool.submit(run_pcgcv2_one,ply_path)
            total_bpp += bpp.result()
            print(f"{ply_path} : bpp {bpp.result()}")
        pool.shutdown()
        print(f"avg bpp: {total_bpp/len(ply_list)}")
    else:
        ply_path = ply_dir
        bpp = run_pcgcv2_one(ply_path)
        print(f"{ply_path} : bpp {bpp}")
