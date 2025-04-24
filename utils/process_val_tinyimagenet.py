import os
import os.path as osp
import pandas
import shutil
import tqdm

data_path = '/home/jeanner211/DATASETS/tiny-imagenet-200'
df = pandas.read_csv(
    osp.join(data_path, 'val', 'val_annotations.txt'),
    sep='\t', header=None)

for folder in df.iloc[:, 1].unique():
    os.makedirs(osp.join(data_path, 'processed-val', folder),
                exist_ok=True)

    for f in df[df.iloc[:,1] == folder].iloc[:, 0]:
        shutil.copyfile(osp.join(data_path, 'val', 'images', f),
                        osp.join(data_path, 'processed-val', folder, f))


