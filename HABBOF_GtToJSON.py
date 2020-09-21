import json
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--root_path", required=True)
args, l = ap.parse_known_args()

HABBOF_path = args.root_path

paths = ["Lab1", "Lab2", "Meeting1", "Meeting2"]

for p in paths:
    datalist = []
    dumpfile = HABBOF_path + "annotations/" + p + ".json"
    for fname in os.listdir(HABBOF_path + p):
        full_file = HABBOF_path + p + "/" + fname
        if(fname[-3:] == "txt"):
            with open(full_file, "r") as f:
                lines = f.read().splitlines()
                for detection in lines:
                    data = {}
                    box = [int(i) for i in detection.split()[1:]]
                    data["image_id"] = fname.replace(".txt", "")
                    data["bbox"] = box
                    data["score"] = 1  # meaningless
                    data["segmentation"] = []  # meaningless
                    datalist.append(data)
    dump_data = {}
    dump_data["annotations"] = datalist
    with open(dumpfile, 'w') as f:
        json.dump(dump_data, f, indent=1)
