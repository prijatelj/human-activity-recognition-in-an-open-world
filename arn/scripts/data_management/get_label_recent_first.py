"""Originally from an ipython session, to be cleaned up,"""
from copy import deepcopy

from arn.models import fine_tune, fine_tune_lit
from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *

kuni = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    #'/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    sample_dirs=BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/features/timesformer/timesformer/',
        batch_col='batch_x3d',
    ),
    #subset=KineticsUnifiedSubset(
    #    kinetics400=KineticsSplitConfig(validate=True),
    #    labels=LabelConfig('label_kinetics400', known=True)
    #),
    #blacklist='../data/kinetics400/blacklists/x3d/blacklist-x3d-k4-train-uid.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)

df = kuni.data.copy(deep=True)
df.drop(columns=['sample_path', 'sample_index'], inplace=True)

df['label_recent_first'] = df['label_kinetics700_2020']
sum(pd.isna(df['label_recent_first']))

# This does not cover all cases of missing k7 label, but decent for.
df_no_k7_test = df[
    ~(
        pd.isna(df.split_kinetics400)
        & pd.isna(df.split_kinetics600)
        & (df.split_kinetics700_2020 == 'test')
    )
]

# For more accuracte confusion matrices of k4 to k6, k4 to k7, and k6 to k7.
df_only_k7_label = df[~pd.isna(df.label_kinetics700_2020)]

# This covers all missing k7 labels where there is a k4 OR k6 label.
df_no_k7_label = df[
    pd.isna(df.label_kinetics700_2020)
    & (
        ~pd.isna(df.label_kinetics400)
        | ~pd.isna(df.label_kinetics600)
    )
]

"""
df_no_k7_test
df_no_k7_test.label_recent_first.unique()
pd.isna(df_no_k7_test.label_recent_first)
sum(pd.isna(df_no_k7_test.label_recent_first))
261212- sum(pd.isna(df_no_k7_test.label_recent_first))
261212 - sum(pd.isna(df_no_k7_test.label_recent_first))
sum(pd.isna(df_no_k7_test.label_recent_first))
#"""

missing_labels = pd.isna(df_no_k7_test.label_recent_first)

"""
sum(missing_labels)
pd.isna(df_no_k7_test[missing_labels].label_kinetics600)
sum(pd.isna(df_no_k7_test[missing_labels].label_kinetics600))
195491 - sum(pd.isna(df_no_k7_test[missing_labels].label_kinetics600))
195491 - sum(pd.isna(df_no_k7_test[missing_labels].label_kinetics400))
#"""

"""
df[['label_kinetics600', 'label_kinetics700_2020']]
df[['label_kinetics600', 'label_kinetics700_2020']].values
list(df[['label_kinetics600', 'label_kinetics700_2020']])
df[['label_kinetics600', 'label_kinetics700_2020']]
list(df[['label_kinetics600', 'label_kinetics700_2020']].values)

[(i[0], [1]) for i in df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values]
#"""



k7_train = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    #'/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    sample_dirs=BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/features/x3d-maybe/x3d/',
        batch_col='batch_x3d',
    ),
    subset=KineticsUnifiedSubset(
        kinetics700_2020=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kinetics700_2020', known=True)
    ),
    #blacklist='../data/kinetics400/blacklists/x3d/blacklist-x3d-k4-train-uid.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
k600_train = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    #'/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    sample_dirs=BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/features/x3d-maybe/x3d/',
        batch_col='batch_x3d',
    ),
    subset=KineticsUnifiedSubset(
        kinetics600=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kinetics600', known=True)
    ),
    #blacklist='../data/kinetics400/blacklists/x3d/blacklist-x3d-k4-train-uid.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)

k6_label_enc = deepcopy(k600_train.label_enc)
k7_label_enc = deepcopy(k7_train.label_enc)

k400_train = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/features/x3d/',
        batch_col='batch_x3d',
    ),
    KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    #blacklist='../data/kinetics400/blacklists/blacklist-x3d-k4-train-uid.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
k4_label_enc = deepcopy(k400_train.label_enc)

# pop unknown class
#k4_label_enc.pop('unknown')
#k6_label_enc.pop('unknown')
#k7_label_enc.pop('unknown')

df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values
df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values[:, 0]
df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']][~pd.isna(df_no_k7_test['label_kinetics600'])]


# Copy the label encoders in case changes occur.
k4_label_enc = deepcopy(k400_train.label_enc)
k6_label_enc = deepcopy(k600_train.label_enc)
k7_label_enc = deepcopy(k7_train.label_enc)


## Calculate and Save the confusion matrices of k4 -> k6, k4 -> k7, and k6 -> k7
k6_to_k7_conf = np.zeros((601,701))
df_k6_to_k7 = df_only_k7_label[~pd.isna(df_only_k7_label['label_kinetics600'])][['label_kinetics600', 'label_kinetics700_2020']]
np_k6_to_k7 = np.hstack((k6_label_enc.encode(df_k6_to_k7.values[:, 0]).reshape(-1,1), k7_label_enc.encode(df_k6_to_k7.values[:, 1]).reshape(-1,1)))
for (row, col) in np_k6_to_k7:
    k6_to_k7_conf[row, col] += 1
#k6_to_k7_conf
#k6_to_k7_conf.sum()
#df_k6_to_k7
np.save('../data/analysis/kinetics/k6_to_k7_confusion.np', k6_to_k7_conf)
#np.load('../data/analysis/kinetics/k6_to_k7_confusion.np.npy')

k4_to_k7_conf = np.zeros((401,701))
df_k4_to_k7 = df_only_k7_label[~pd.isna(df_only_k7_label['label_kinetics400'])][['label_kinetics400', 'label_kinetics700_2020']]#.fillna('unknown')
#df_k4_to_k7
np_k4_to_k7 = np.hstack((k4_label_enc.encode(df_k4_to_k7.values[:, 0]).reshape(-1,1), k7_label_enc.encode(df_k4_to_k7.values[:, 1]).reshape(-1,1)))
#np_k4_to_k7
for (row, col) in np_k4_to_k7:
    k4_to_k7_conf[row, col] += 1
#k4_to_k7_conf
#k4_to_k7_conf.sum()
np.save('../data/analysis/kinetics/k4_to_k7_confusion', k4_to_k7_conf)

k4_to_k6_conf = np.zeros((401,601))
df_k4_to_k6 = df_only_k7_label[['label_kinetics400', 'label_kinetics600']][~pd.isna(df_only_k7_label['label_kinetics400'])].fillna('unknown')
np_k4_to_k6 = np.hstack((k4_label_enc.encode(df_k4_to_k6.values[:, 0]).reshape(-1,1), k6_label_enc.encode(df_k4_to_k6.values[:, 1]).reshape(-1,1)))
#np_k4_to_k6
for (row, col) in np_k4_to_k6:
    k4_to_k6_conf[row, col] += 1
#k4_to_k6_conf.sum()
#k4_to_k6_conf == k6_to_k7_conf
#k4_to_k6_conf == k4_to_k7_conf
np.save('../data/analysis/kinetics/k4_to_k6_confusion', k4_to_k6_conf)

unk_argmax_k6_to_k7 = np.where(k6_to_k7_conf.argmax(1) == 0)[0]

k_argmax_select_unks_argmax = k6_to_k7_conf[:, 1:].argmax(1)[unk_argmax_k6_to_k7] + 1

k4_idx_not_in_k7 = list(pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k6_label_enc)))].index)

k4_series_not_in_k7 = pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k7_label_enc)))]


## Fill in the k7 NaNs that have k6 labels
missing_labels = pd.isna(df_no_k7_label.label_recent_first)

df_k6_to_k7 = df_no_k7_label[~pd.isna(df_no_k7_label['label_kinetics600'])]
missing_idx_k6_recent = df_no_k7_label[missing_labels].index
missing_idx_k6_recent = missing_idx_k6_recent[missing_idx_k6_recent.isin(df_k6_to_k7.index)]

#df_no_k7_label.loc[missing_idx_k6_recent, 'label_recent_first'] = df_k6_to_k7.loc[missing_idx_k6_recent, 'label_kinetics600']
df_no_k7_label.loc[missing_idx_k6_recent, 'label_recent_first'] = df_no_k7_label.loc[missing_idx_k6_recent, 'label_kinetics600']

# Update mask of missing labels
missing_labels = pd.isna(df_no_k7_label.label_recent_first)
missing_idx_k4_recent = df_no_k7_label[missing_labels].index

## Fill in the remaining k7 NaNs that have k4 labels
#df_k4_to_k7 = df_no_k7_label[~pd.isna(df_no_k7_label['label_kinetics400'])]
#df_no_k7_label.loc[missing_idx_k4_recent, 'label_recent_first'] = df_k4_to_k7.loc[missing_idx_k4_recent, 'label_kinetics400']

df_no_k7_label.loc[missing_idx_k4_recent, 'label_recent_first'] = df_no_k7_label.loc[missing_idx_k4_recent, 'label_kinetics400']


#df.to_csv('/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched_recent_first.csv', index=False)


remaps = {
    'balloon blowing': 'inflating balloons',
    'dying hair': 'dyeing hair',
    'garbage collecting': 'person collecting garbage',
    'making bed': 'making the bed',
    'strumming guitar': 'playing guitar',
    'tying tie': 'tying necktie',
}


# This covers all missing k7 labels where there is a k4 OR k6 label.
df_no_k7_label = df[
    pd.isna(df.label_kinetics700_2020)
    & (
        ~pd.isna(df.label_kinetics400)
        | ~pd.isna(df.label_kinetics600)
    )
]
missing_labels = pd.isna(df_no_k7_label.label_recent_first)

# K6 first
df_k6_to_k7 = df_no_k7_label[~pd.isna(df_no_k7_label['label_kinetics600'])]
missing_idx_k6_recent = df_no_k7_label[missing_labels].index
missing_idx_k6_recent = missing_idx_k6_recent[missing_idx_k6_recent.isin(df_k6_to_k7.index)]


# K4 next
df_no_k7_label.loc[missing_idx_k6_recent, 'label_recent_first'] = df_no_k7_label.loc[missing_idx_k6_recent, 'label_kinetics600']
missing_labels = pd.isna(df_no_k7_label.label_recent_first)
missing_idx_k4_recent = df_no_k7_label[missing_labels].index

df_no_k7_label.loc[missing_idx_k4_recent, 'label_recent_first'] = df_no_k7_label.loc[missing_idx_k4_recent, 'label_kinetics400']

# Remap those in k4 that need remapped.
for k, v in remaps.items():
    mask = df_no_k7_label.label_kinetics400[missing_labels] == k
    mask = df_no_k7_label[missing_labels][mask].index
    print(mask)
    df_no_k7_label.loc[mask, 'label_recent_first'] = v
