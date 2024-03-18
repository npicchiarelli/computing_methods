'''Unit ests for utils module.
'''

import pathlib

import awkward as ak
import librosa
from loguru import logger
import numpy as np
import pandas as pd
import torch

from utils.utils import import_audio_dataset, file_names_map, encode_file_names, \
    pad_awkward_array, audio_tokenizer, downsample_tensor, get_file_list

def test_get_file_list():
    """Unit test for get_file_list"""
    
    file_list = get_file_list('/home/npic/computing_methods/coding/cmepda_exam/tests/test_dir/ravdess', '.wav')
    assert len(file_list) == 7
    assert all([path.suffix == '.wav' for path in file_list])

def test_import_audio_dataset():
    """Unit test for import_audio_dataset"""

    array, sr = import_audio_dataset('/home/npic/computing_methods/coding/cmepda_exam/tests/test_dir/ravdess', '.wav', verbose = False)
    assert sr == 48000
    assert isinstance(array, ak.highlevel.Array)
    assert max([len(np.asarray(np.ravel(ts))) for ts in array]) == 187388
    assert min([len(np.asarray(np.ravel(ts))) for ts in array]) == 158558

def test_file_names_map():
    """Unit test for file_names_map"""

    separator = "_"
    encodict = {
                "a": {'1': 'pippo'},
                "c": {'DFA': "pluto"},
                "b": {'ANG': 'topolino'},
                "d": {'XX': 'FDS'},
            }

    path = 'test_dir/crema/1001_DFA_ANG_XX.wav'
    file = [pathlib.Path(path)]
    df = file_names_map(file, separator, encodict)

    assert list(df.columns) == ['a', 'c', 'b', 'd', 'filename']
    assert all(df.to_numpy().squeeze() == ['1001', 'pluto', 'topolino',
                                           'FDS', '1001_DFA_ANG_XX.wav'])
    assert df["filename"][0] == path[15:]

def test_encode_file_names():
    """Unit test for encode_file_names"""

    crema_list = get_file_list('./test_dir/crema','.wav')
    ravdess_list = get_file_list('./test_dir/ravdess', '.wav')
    assert len(encode_file_names(crema_list, 'crema').columns) == 5
    assert len(encode_file_names(ravdess_list, 'ravdess').columns) == 9

def test_pad_awkward_array():
    """Unit test for pad_awkward_array"""

    x, sr = import_audio_dataset('./test_dir/crema', '.wav', verbose=False)
    x_padded = pad_awkward_array(x).to_numpy().squeeze()
    assert x_padded.shape[0] == max((len(np.asarray(np.ravel(ts))) for ts in x))
    x_t = np.array(x[0]).squeeze()
    assert np.allclose(x_padded[0, :len(x_t)], x_t)

def test_audio_tokenizer():
    """Unit test for audio_tokenizer"""

    x, sr = import_audio_dataset('./test_dir/crema', '.wav', verbose=False)
    x_padded = torch.tensor(pad_awkward_array(x)).squeeze()
    x_tok, nt, st,  length = audio_tokenizer(x_padded, sr, 27.5, return_sizes=True, verbose = False)
    assert x_tok.shape[0] == x_padded.shape[0]
    assert x_tok.shape[1] == nt
    assert x_tok.shape[1]*x_tok.shape[2] == length
    assert torch.allclose(x_padded[0,:st], x_tok[0,0,:])

def test_downsample_tensor():
    """Unit test for downsample_tensor"""

    for ds in ['crema', 'ravdess']:
        x, sr = import_audio_dataset(f'./test_dir/{ds}', '.wav', verbose=False)
        x_padded = torch.tensor(pad_awkward_array(x)).squeeze()
        target_sr = 16000
        x_dn = downsample_tensor(x_padded, sr, target_sr)
        if sr == target_sr:
            assert torch.allclose(x_dn, x_padded)
        else:
            print((x_dn.shape[1]/x_padded.shape[1])**-1)
            print(float(target_sr/sr)**-1)
            assert np.allclose(x_padded.shape[1]/x_dn.shape[1], float(sr/target_sr), atol = 1e-2)
