import pandas as pd
train = pd.read_csv("train_split.csv")
test = pd.read_csv("test_split.csv")
print(len(train), len(test))
print(train["median_Tm"].describe())