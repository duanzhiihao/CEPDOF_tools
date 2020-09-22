import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--root_path", required=True)
args, l = ap.parse_known_args()

MW_path = args.root_path

for vidname in os.listdir(MW_path):
    day = vidname.split("-")[-1]
    framepath = MW_path + vidname + "/img1/"
    for im in os.listdir(framepath):
        dest = "Mar" + day + "_" + im[:-4] + ".jpg"
        os.rename(framepath+im, MW_path + dest)
