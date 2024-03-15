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
from tqdm import tqdm

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

def import_audio_dataset(dir_path: str, suffix, verbose = True):
    """Returns the audio dataset as an awkward array
       along with the sample rate of the audio files. 

    This will accept as input a path to the directory where files are located
    along with the extension of the files.
    Please note this will work only with codecs supported by Soundfile.

    Arguments
    ---------
    dir_path : string
        The path to the directory.
    suffix : string
        The files extension i.e. '.wav', '.mp3' etc
    verbose : bool
        if it is True, progress bars and info are displayed
    Return
    ------
    The dataset array : awkward.Array
    The sample rate in Hz: float 
    """
    dataset_folder = pathlib.Path(dir_path)

    file_list = get_file_list(dataset_folder, suffix)

    array = []
    if verbose:
        logger.info(f"Loading {len(file_list)} files to list...")
    for path in tqdm(file_list, disable = not verbose):
        librosa_audio_segment, sr = librosa.load(path, sr=None)
        array.append([librosa_audio_segment])
        b = ak.ArrayBuilder()

    if verbose:
        logger.info(f"Loading {len(file_list)} files to awkward array...")
    for path in tqdm(file_list, disable = not verbose):
        librosa_audio_segment, sr = librosa.load(path, sr=None)
        b.begin_list()
        b.begin_list()
        for value in librosa_audio_segment:
            b.real(value)
        b.end_list()
        b.end_list()
    array = b.snapshot()

    return array, sr

def file_names_map(files: list, separator: str, enc_dict: dict):
    """Encodes the information contained in the file names.
    Returns a pandas dataframe containing all the categorical variables
    as encoded in file names.
    
    Arguments
    ---------
    files : list[Path]
        The list of file paths as returned by get_file_list.
    separator : str
        The separator used in the file names.
    encoding : dict or string
        The encoding dictionary.

    Return
    ------
    The pandas dataframe containing the encoded categorical variables: pd.DataFrame
    """

    df = []
    file_names = []
    for path in files:
        stem = path.stem.split(separator)
        file_names.append(path.name)
        df.append(stem)
    df = pd.DataFrame(df)
    df.columns = list(enc_dict.keys())

    df = df.replace(enc_dict)

    df["filename"] = file_names
    return df

class DatasetError(Exception):
    """Class for error in
    dataset selection"""

def encode_file_names(files: list, encoding, **kwargs):
    """Extends file_names_map functionalities adding specifical features
    for the two datasets used in the project: ravdess and crema.

    Arguments
    ---------
    files : list[Path] or str
        The list of file paths as returned by get_file_list.    
    encoding : dict or string
        The encoding dictionary or the dataset name for the two 
        built in dataset encodings: 'ravdess', 'crema'.
    **kwargs: separator : str
        The separator used in the file names

    Return
    ------
    The pandas dataframe containing the encoded categorical variables: pd.dataframe

    Raises
    ------
    NoDatasetError
    if a string is passed as input and it is not the name of a 
    built in dataset: 'ravdess' or 'crema'.
    TypeError
    if "encoding" is not of type str or dict
    """

    if isinstance(encoding, str):
        if encoding == 'ravdess':
            if str(files[0].name)[2] == "-":
                separator = "-"
                categorical_features_names = {
                    "modality": {"01": "full-AV", "02": "video-only", "03": "audio-only"}, 
                    "vocal_channel": {"01": "speech", "02": "song"},
                    "emotion": {"01" : "neutral",
                                "02" : "calm",
                                "03" : "happy",
                                "04" : "sad",
                                "05" : "angry",
                                "06" : "fearful",
                                "07" : "disgust",
                                "08" : "surprised"},
                    "emotional_intensity": {"01" : "normal", "02" : "strong"},
                    "statement": {"01" : "Kids are talking by the door",
                                  "02" : "Dogs are sitting by the door"},
                    "repetition": {"01" : "1st", "02" : "2nd"},
                    "actor": {str(i).zfill(2): str(i).zfill(2) for i in range(1, 25)}
                }
                df = file_names_map(files, separator, categorical_features_names)
                df["sex"] = ["F" if i % 2 == 0 else "M" for i in df["actor"].astype(int)]
            else:
                raise DatasetError(f'Wrong separator in file names:'
                                   f'{str(files[0].name)[2]} was given but - was expected.')
        elif encoding == 'crema':
            if str(files[0].name)[4] == "_":
                separator = "_"
                categorical_features_names = {
                    "actor": {f'10{str(i).zfill(2)}': str(i).zfill(2) for i in range(1, 92)},
                    "statement": {'IEO': "It's eleven o'clock",
                                  'TIE': "That is exactly what happened",
                                  'IOM': "I'm on my way to the meeting",
                                  'IWW': "I wonder what this is about",
                                  'TAI': "The airplane is almost full",
                                  'MTI': "Maybe tomorrow it will be cold",
                                  'IWL': "I would like a new alarm clock",
                                  'ITH': "I think I have a doctor's appointment",
                                  'DFA': "Don't forget a jacket",
                                  'ITS': "I think I've seen this before",
                                  'TSI': "The surface is slick",
                                  'WSI': "We'll stop in a couple of minutes"},
                    "emotion": {'ANG': 'anger', 'DIS': 'disgust', 'FEA': 'fear',
                                'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'},
                    "emotional_intensity": {'LO': 'Low', 'MD': 'Medium',
                                            'HI': 'High',
                                            'XX': 'Unspecified'},
                }
                df = file_names_map(files, separator, categorical_features_names)
            else:
                raise  DatasetError(f'Wrong separator in file names:'
                                    f'{str(files[0].name)[4]} was given but - was expected.')
        else:
            raise DatasetError('No dataset encoding for the selected dataset name!')

    elif isinstance(encoding, dict):
        df = file_names_map(files, kwargs["separator"], encoding)

    else:
        raise TypeError('The input encoding must be of type str ("ravdess" or "crema") or dict.')

    return df

def pad_awkward_array(x: ak.Array, nan_value = 0.):
    """Takes as input a 3D awkward array,
    returns a 3D awkward.Array padded with nan_value
    to have all the rows of the same length.

    Arguments
    ---------
    x : ak.Array
        The array to pad.   
    nan_value : float
        the value used to pad 

    Return
    ---------
    The padded array : ak.Array
    """

    maximum = 0
    for ts in x:
        length = len(np.asarray(np.ravel(ts)))
        if length > maximum:
            maximum = length

    return ak.fill_none(ak.pad_none(x, maximum, axis=2, clip=True), value=nan_value)
