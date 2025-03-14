import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tarfile
from pathlib import Path
import urllib.request
from zlib import crc32

# loading and spliting our data into train set and test set
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True , exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url , tarball_path)
        with tarfile.open(tarball_path) as housing_tarball :
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing = load_housing_data()

def ShuffleAndSplitData(data , testRatio):
    shuffledIndices = np.random.permutation(len(data))
    test_set_size = int(len(data) * testRatio)
    testIndices = shuffledIndices[:test_set_size]
    trainIndices = shuffledIndices[test_set_size:]
    return data.iloc[trainIndices] ,data.iloc[testIndices]

# the test set will be 20% of the data
ShuffleAndSplitData(housing , 0.2)

# adding unique ids to the data

def is_id_in_test_set(id , testRatio):
    return crc32(np.int64(id)) < testRatio * 2**32

def split_data_with_id_hash(data , test_ratio , id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : is_id_in_test_set(id_ , test_ratio))
    return data.loc[~in_test_set] , data.loc[in_test_set]
