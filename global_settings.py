from pathlib import Path
import os


# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    DRIVE_PATH = "/Volumes/Sumsung_1T/Projects/VAE"
    DATA_PATH = os.path.join(DRIVE_PATH, "data")
    OUTPUT_PATH = os.path.join(DRIVE_PATH, "output")
else:
    DATA_PATH = os.path.join(DESKTOP_PATH, "data")
    OUTPUT_PATH = os.path.join(DESKTOP_PATH, "output")
LOG_PATH = os.path.join(OUTPUT_PATH, "log")
