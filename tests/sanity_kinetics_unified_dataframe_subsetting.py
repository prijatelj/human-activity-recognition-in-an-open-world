# A copy paste of ipython interpretter history of testing the kinetics unified
# subsetting to that of the src kinetics 400 labels for train and validate.

from arn.models import fine_tune_lit
from arn.models.fine_tune import get_residual_map
from arn.data.kinetics_unified import *
from arn.data.kinetics_unified import *
from arn.models import fine_tune_lit
from arn.models.fine_tune import get_residual_map
from arn.data.kinetics_unified import *
import pandas as pd
k400 = KineticsUnifiedFeatures(
    '/home/prijatelj/workspace/research/osr/repos/har/data/analysis/kinetics/kinetics_400_600_700_2020.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    KineticsRootDirs(
        kinetics400_dir='x3d-features-k400/',
        root_dir='/mnt/hdd/workspace/research/osr/har/kitware/'
    ),
    #KineticsUnifiedSubset(
    #    kinetics400=KineticsSplitConfig(train=True),
    #    labels=LabelConfig('label_kinetics400', known=True)
    #),
    #blacklist='../data/kinetics400/x3d-k400-blacklist-train-file.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    #log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
df = pd.read_csv(
    '/home/prijatelj/workspace/research/osr/repos/har/data/analysis/kinetics/kinetics_400_600_700_2020.csv',
                dtype={
                'label_kinetics400': str,
                'label_kinetics600': str,
                'label_kinetics700_2020': str,
                'split_kinetics400': str,
                'split_kinetics600': str,
                'split_kinetics700_2020': str,
            },
)
df
k400.data
k400.data.columns
df.columns
k400.data.sample_index
k400.data.sample_index == k400.data.index
(k400.data.sample_index == k400.data.index).all()
(k400.data.sample_index == df.index).all()
k400.data
k400.data.sample_path
k400.data.sample_path.iloc[0]
k400.data.youtube_id.iloc[0]
df.youtube_id.iloc[0]
(df.youtube_id == k400.data.youtube_id).all()
(df.label_kinetics400 == k400.data.label_kinetics400).all()
(df.label_kinetics400 == k400.data.label_kinetics400)
df.fillna('NaN')
k400.data.fillna('NaN')
(df.label_kinetics400 == k400.data.label_kinetics400)
df.label_kinetics400
k4df = k400.data.fillna('NaN')
k4df
dfnan = df.fillna('NaN')
(dfnan.label_kinetics400 == k4df.label_kinetics400)
(dfnan.label_kinetics400 == k4df.label_kinetics400).all()
k400 = KineticsUnifiedFeatures(
    '/home/prijatelj/workspace/research/osr/repos/har/data/analysis/kinetics/kinetics_400_600_700_2020.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    KineticsRootDirs(
        kinetics400_dir='x3d-features-k400/',
        root_dir='/mnt/hdd/workspace/research/osr/har/kitware/'
    ),
    KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    #blacklist='../data/kinetics400/x3d-k400-blacklist-train-file.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    #log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
k400
k400.data
df
k4df
kdf = k4df
kdf
k4df = pd.read_csv(
    '/home/prijatelj/workspace/research/osr/repos/har/data/kinetics400/train.csv',
                dtype={
                'label_kinetics400': str,
                'label_kinetics600': str,
                'label_kinetics700_2020': str,
                'split_kinetics400': str,
                'split_kinetics600': str,
                'split_kinetics700_2020': str,
            },
)
k4df
k4df.columns
k400.data
k400.data.label_kinetics400 == k4df
(k400.data.label_kinetics400 == k4df.label)
(k400.data.label_kinetics400 == k4df.label).all()
(k400.data.index == k4df.index).all()
(k400.data.sample_index == k4df.index).all()
k4df_val = pd.read_csv(
    '/home/prijatelj/workspace/research/osr/repos/har/data/kinetics400/validate.csv',
                dtype={
                'label_kinetics400': str,
                'label_kinetics600': str,
                'label_kinetics700_2020': str,
                'split_kinetics400': str,
                'split_kinetics600': str,
                'split_kinetics700_2020': str,
            },
)
k4df_val
k400_val = KineticsUnifiedFeatures(
    '/home/prijatelj/workspace/research/osr/repos/har/data/analysis/kinetics/kinetics_400_600_700_2020.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    KineticsRootDirs(
        kinetics400_dir='x3d-features-k400/',
        root_dir='/mnt/hdd/workspace/research/osr/har/kitware/'
    ),
    KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(validate=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    #blacklist='../data/kinetics400/x3d-k400-blacklist-train-file.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    #log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
k400_val.df
k4df_val
k400_val.data
(k400_val.data.label_kinetics400 == k4df_val.label).all()
(k400_val.data.label_kinetics400.values == k4df_val.label.values).all()
(k400_val.data.youtube_id.values == k4df_val.youtube_id.values).all()

# At this point, the k400 test and train are correct when using subset for
# train and validate alone, as we use in the experiments.

# What about after blacklisting and whitelisting non-existent files?
k400_train = KineticsUnifiedFeatures(
    '/home/prijatelj/workspace/research/osr/repos/har/data/analysis/kinetics/kinetics_400_600_700_2020.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    KineticsRootDirs(
        kinetics400_dir='x3d-features-k400/',
        root_dir='/mnt/hdd/workspace/research/osr/har/kitware/'
    ),
    KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    blacklist='../data/kinetics400/x3d-k400-blacklist-train-file.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    #log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
k400_train.data
k400.data
len(k400.data)
len(k400.data) - len(k400_train)

blacklist = pd.read_csv('../data/kinetics400/x3d-k400-blacklist-train-file.log')
blacklist
blacklist = pd.read_csv('../data/kinetics400/x3d-k400-blacklist-train-file.log', header=None)
blacklist
for i in blacklist[0]: pass
split = []
for i in blacklist[0]:
    split.append(i.split('_'))
split
split = []
for i in blacklist[0]:
    print(i.rpartition('_'))
    break
split = []
for i in blacklist[0]:
    rem, _, end = i.rpartition('_')
    yid, _, start = rem.rpartition('_')
    split.append([yid, start, end])
split
k400_train
k400_train.data
k400_train.data.columns
k400_train.data['youtube_id']
k400_train.data['time_start']
split = []
for i in blacklist[0]:
    rem, _, end = i.rpartition('_')
    yid, _, start = rem.rpartition('_')
    split.append([yid, int(start), int(end)])
split
blacklist
blacklist_split = pd.DataFrame(split, columns=['youtube_id', 'time_start', 'time_end'])
blacklist_split
k400_train.data[['youtube_id', 'time_start', 'time_end']]
k400.data[['youtube_id', 'time_start', 'time_end']]
for i in blacklist_split.loc:
    print(i)
blacklist_split
for i in blacklist_split.iloc:
    print(i)
k400.data
k400.youtube_id.isin(blacklist_split.youtube_id)
k400.data.youtube_id.isin(blacklist_split.youtube_id)
blacklist_mask = ~(
    k400.data.youtube_id.isin(blacklist_split.youtube_id)
    & k400.data.time_start.isin(blacklist_split.time_start)
    & k400.data.time_end.isin(blacklist_split.time_end)
)
blacklist_mask
k400.data[blacklist_mask]
k400.data[blacklist_mask] == k400_train.data
(k400.data[blacklist_mask] == k400_train.data)all()
(k400.data[blacklist_mask] == k400_train.data).all()
((k400.data[blacklist_mask] == k400_train.data) | pd.isna(k400_train.data)).all()

# From this the blacklist for train is performed correctly.  could perform the
# same check for validation, but at this point the train labels should be
# correct, thus if data is good and the predictor is good, the predictor should
# be getting around similar performance to the X3D.
