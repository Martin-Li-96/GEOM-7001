import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

data=np.load("./labeled_images.npz")
x=data["x"]
y=data["y"]
sss=StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
# for i, (train_index, test_index) in enumerate(sss.split(x, y)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}")
#     print(f"  Test:  index={test_index}")

for index,(train_index, test_index) in enumerate(sss.split(x,y)):
    np.savez("Dataset_whole_90_10",train_x=x[train_index],train_y=y[train_index],test_x=x[test_index],test_y=y[test_index])
