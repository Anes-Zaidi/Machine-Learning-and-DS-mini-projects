import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tarfile
from pathlib import Path
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder , StandardScaler

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

# spliting & shuflling the data into training and test sets based on the income category

housing ["income_cat"] = pd.cut(housing["median_income"] , bins=[0 ,1.5 , 3 , 4.5 , 6 , np.inf] , labels=[i+1 for i in range(5)])

# test set will be 20% of the data
strat_trainSet , strat_testSet = train_test_split(housing , test_size=0.2 , stratify=housing["income_cat"] , random_state=42)

# removing income category now since we wont use it anymore
for sett in( strat_trainSet , strat_testSet) :
    sett.drop("income_cat" , axis = 1 ,inplace = True)

#preparing the data 
housing = strat_trainSet.drop("median_house_value" , axis = 1 )
housing_lables = strat_trainSet["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])

x = imputer.fit(housing_num)

housing_tr = pd.DataFrame(housing_num , columns=housing_num.columns , index=housing_num.index)

# transforming the ocean proximity categorical data into numerical data
housing_cat = housing["ocean_proximity"]

ord_enc = OrdinalEncoder()
onehot_enc = OneHotEncoder()

housing_cat_encoded = ord_enc.fit_transform(housing_cat.to_frame())
housing_cat_1hot = onehot_enc.fit_transform(housing_cat.to_frame())

# scaling the data using the standardScaler
std = StandardScaler()
housing_num_scaled  = std.fit_transform(housing_num)



