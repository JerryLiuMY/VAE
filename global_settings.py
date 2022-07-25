from pathlib import Path
import numpy as np
import pandas as pd
import os


# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    SEAGATE_PATH = "/Volumes/Sumsung_1T/Projects/ML_Prediction"
    DATA_PATH = os.path.join(SEAGATE_PATH, "shared")
    OUTPUT_PATH = os.path.join(SEAGATE_PATH, "result")
else:
    DATA_PATH = os.path.join(DESKTOP_PATH, "shared")
    OUTPUT_PATH = os.path.join(DESKTOP_PATH, "result")
LOG_PATH = os.path.join(OUTPUT_PATH, "log")


# load necessary files
# trddt_all: intersection between X and y
# cusip_all: intersection between union(X) and union(y)
# cusip_all: cusip_all with match to sic code

trddt_all = np.asarray(pd.read_pickle(os.path.join(DATA_PATH, "trddt_all.pkl")))
cusip_all = np.asarray(pd.read_pickle(os.path.join(DATA_PATH, "cusip_all.pkl")))
cusip_sic = pd.read_csv(os.path.join(DATA_PATH, "cusip_sic.txt"), delim_whitespace=True)
date0_min = "2010-03-31"
date0_max = "2022-06-14"


# TODO
# Set plotting legend
# US daily alpha / ETF
# Chinese daily alpha / stock index option
# Crypto statistical arbitrage
# Ultra high frequency

# modify the sampling scheme
# plotting sync x_label
# mask input with nan values -- drop directly
# check for NaN values in correlation
# model weights
# informer/autoformer
