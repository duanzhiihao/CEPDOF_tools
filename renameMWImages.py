import os

MW_path = "E:/MWAll/MW-R/MW-R/MW-R/"

for vidname in os.listdir(MW_path):
    day = vidname.split("-")[-1]
    framepath = MW_path + vidname + "/img1/"
    for im in os.listdir(framepath):
        dest = "Mar" + day + "_" + im[:-4] + ".jpg"
        os.rename(framepath+im, MW_path + dest)
