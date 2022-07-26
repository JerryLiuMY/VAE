import torch
from pathlib import Path
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    DRIVE_PATH = "/Volumes/Sumsung_1T/Projects/VAE"
    DATA_PATH = os.path.join(DRIVE_PATH, "data")
    OUTPUT_PATH = os.path.join(DRIVE_PATH, "output")
else:
    DATA_PATH = os.path.join(DESKTOP_PATH, "data")
    OUTPUT_PATH = os.path.join(DESKTOP_PATH, "output")


# make directories
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
