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
    imlist = []
    dumpfile = HABBOF_path + "annotations/" + p + ".json"
    for fname in os.listdir(HABBOF_path + p):
        full_file = HABBOF_path + p + "/" + fname
        if(fname[-3:] == "txt"):
            imdata = {}
            imdata["file_name"] = fname.replace(".txt", ".jpg")
            imdata["id"] = fname.replace(".txt", "")
            imdata["width"] = 2048
            imdata["height"] = 2048
            imlist.append(imdata)
            with open(full_file, "r") as f:
                lines = f.read().splitlines()
                for detection in lines:
                    data = {}
                    box = [float(i) for i in detection.split()[1:]]
                    data["image_id"] = fname.replace(".txt", "")
                    data["category_id"] = 1
                    data["iscrowd"] = 0
                    data["area"] = box[2] * box[3]
                    data["bbox"] = box
                    data["segmentation"] = []
                    datalist.append(data)
    dump_data = {}
    dump_data["annotations"] = datalist
    dump_data["images"] = imlist
    dump_data["categories"] = [
        {"id": 1, "name": "person", "supercategory": "person"}]
    with open(dumpfile, 'w') as f:
        json.dump(dump_data, f, indent=1)
