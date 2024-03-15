'''Utilities for audio dataset import and preparation for tasks
   involving transformers. 
'''

import pathlib

import awkward as ak
import librosa
from loguru import logger
import numpy as np
import pandas as pd
import scipy
import torch

def get_file_list(dir_path: str, suffix: str):
    """Returns a sorted list containing the paths to all the files contained in 'dir_path'
     with 'suffix' as extension. 

    This will accept as input a path to the directory where files are located
    in the form of '/directory/' and the suffix.

    Arguments
    ---------
    path : string
        The path to the directory.
    suffix : string
        The files extension i.e. '.wav', '.pdf' etc

    Return
    ------
        The list of relative file paths : list[Path]
    """
    dataset_folder = pathlib.Path(dir_path)

    file_list = list(dataset_folder.iterdir())
    for i, path in enumerate(file_list):
        if path.suffix != suffix:
            file_list.pop(i)
    file_list = sorted(file_list)

    return file_list
