"""Originally from an ipython session, to be cleaned up,"""

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
df = kuni.data.copy
df.columns
df = kuni.data.copy(deepcopy=True)
df = kuni.data.copy(deep=True)
df.columns
df.drop(columns=['sample_path', 'sample_index'], inplace=True)
df
df.columns
df['label_recent_first'] = df['label_kinetics700_2020']
df.columns
df.columns
df
sum(pd.isna(df['label_recent_first']))
df_no_k7_test = df[~(pd.isna(df.split_kinetics400) & pd.isna(df.split_kinetics600) & (df.split_kinetics700_2020 == 'test'))]
df_no_k7_test
df_no_k7_test.label_recent_first.unique()
pd.isna(df_no_k7_test.label_recent_first)
sum(pd.isna(df_no_k7_test.label_recent_first))
261212- sum(pd.isna(df_no_k7_test.label_recent_first))
261212 - sum(pd.isna(df_no_k7_test.label_recent_first))
sum(pd.isna(df_no_k7_test.label_recent_first))
df_no_k7_test
missing_labels = pd.isna(df_no_k7_test)
missing_labels
missing_labels = pd.isna(df_no_k7_test.label_recent_first)
missing_labels
sum(missing_labels)
missing_labels
missing_labels
ipython
df_no_k7_test[missing_labels)
df_no_k7_test[missing_labels]
df_no_k7_test[missing_labels].label_kinetics600
pd.isna(df_no_k7_test[missing_labels].label_kinetics600)
sum(pd.isna(df_no_k7_test[missing_labels].label_kinetics600))
195491 - sum(pd.isna(df_no_k7_test[missing_labels].label_kinetics600))
195491 - sum(pd.isna(df_no_k7_test[missing_labels].label_kinetics400))
df
df[['label_kinetics600', 'label_kinetics700_2020']]
df[['label_kinetics600', 'label_kinetics700_2020']].values
list(df[['label_kinetics600', 'label_kinetics700_2020']])
df[['label_kinetics600', 'label_kinetics700_2020']]
list(df[['label_kinetics600', 'label_kinetics700_2020']].values)
pairs = [(x[0], x[1]) for xdf[['label_kinetics600', 'label_kinetics700_2020']].values]
pairs = [(x[0], x[1]) for df[['label_kinetics600', 'label_kinetics700_2020']].values]
pairs = [(x[0], x[1]) for df[['label_kinetics600', 'label_kinetics700_2020']].values]
df[['label_kinetics600', 'label_kinetics700_2020']].values
df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values
[i for i in df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values]
[(i[0], [1]) for i in df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values]
k6_k7_conf = np.zeros((600,700))
k6_k7_conf
k6_k7_conf.shape
k7_train
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
k600_train
k600_train.label_enc
list(k600_train.label_enc)
from copy import deepcopy
k6_label_enc = deepcopy(k600_train.label_enc)
k7_label_enc = deepcopy(k7_train.label_enc)
k400 = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/features/x3d/',
        batch_col='batch_x3d',
    ),
    #KineticsUnifiedSubset(
    #    kinetics400=KineticsSplitConfig(test=True),
    #    labels=LabelConfig('label_kinetics400', known=True)
    #),
    #blacklist='../data/kinetics400/blacklists/blacklist-x3d-k4-train-uid.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
k400
k400
k400
k400
k4_label_enc = deepcopy(k400.label_enc)
k400
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
k4_label_enc
list(k4_label_enc)
len(k4_label_enc)
len(k6_label_enc)
len(k7_label_enc)
k4_label_enc.pop('unknown')
len(k4_label_enc)
k4_label_enc.unknown_idx
k4_label_enc.unknown_key
k6_label_enc.pop('unknown')
k7_label_enc.pop('unknown')
k6_k7_conf
df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values
df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']].values[:, 0]
df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']][~pd.isna(df_no_k7_test['label_kinetics600'])]
k4_label_enc = deepcopy(k400_train.label_enc)
k6_label_enc = deepcopy(k600_train.label_enc)
k7_label_enc = deepcopy(k7_train.label_enc)
k6_k7_conf = np.zeros((601,701))
k6_to_k7_conf = np.zeros((601,701))
del k6_k7_conf
k6_to_k7_conf
k6_to_k7_conf.shape
df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']][~pd.isna(df_no_k7_test['label_kinetics600'])]
df_k6_to_k7 = df_no_k7_test[['label_kinetics600', 'label_kinetics700_2020']][~pd.isna(df_no_k7_test['label_kinetics600'])]
df_k6_to_k7
df_k6_to_k7.values
df_k6_to_k7.values[:, 0]
k6_label_enc.encode(df_k6_to_k7.values[:, 0])
k7_label_enc.encode(df_k6_to_k7.values[:, 1])
pd.isna(df_k6_to_k7)
df_k6_to_k7.fillna('unknown')
df_k6_to_k7 = df_k6_to_k7.fillna('unknown')
df_k6_to_k7
k6_label_enc.encode(df_k6_to_k7.values[:, 0]).reshape(-1,1)
k7_label_enc.encode(df_k6_to_k7.values[:, 1]).reshape(-1,1)
np_k6_to_k7 = np.hstack((k6_label_enc.encode(df_k6_to_k7.values[:, 0]).reshape(-1,1), k7_label_enc.encode(df_k6_to_k7.values[:, 1]).reshape(-1,1)))
np_k6_to_k7
np_k6_to_k7.shape
k6_label_enc.decode(np_k6_to_k7[0,0])
k6_label_enc.decode(np_k6_to_k7[0,0:2])
k6_label_enc.decode(np_k6_to_k7[0:2,0])
df_k6_to_k7
k6_label_enc.decode(np_k6_to_k7[0:4,0])
np_k6_to_k7
k6_to_k7_conf
for (row, col) in np_k6_to_k7:
    k6_to_k7_conf[row, col] += 1
k6_to_k7_conf
k6_to_k7_conf.sum()
df_k6_to_k7
np.save('../data/analysis/kinetics/k6_to_k7_confusion.np', k6_to_k7_conf)
np.load('../data/analysis/kinetics/k6_to_k7_confusion.np.npy')
k4_to_k7_conf = np.zeros((401,701))
df_k4_to_k7 = df_no_k7_test[['label_kinetics400', 'label_kinetics700_2020']][~pd.isna(df_no_k7_test['label_kinetics400'])].fillna('unknown')
df_k4_to_k7
np_k4_to_k7 = np.hstack((k4_label_enc.encode(df_k4_to_k7.values[:, 0]).reshape(-1,1), k7_label_enc.encode(df_k4_to_k7.values[:, 1]).reshape(-1,1)))
np_k4_to_k7
for (row, col) in np_k4_to_k7:
    k4_to_k7_conf[row, col] += 1
k4_to_k7_conf
k4_to_k7_conf.sum()
np.save('../data/analysis/kinetics/k4_to_k7_confusion', k4_to_k7_conf)
k4_to_k6_conf = np.zeros((401,601))
df_k4_to_k6 = df_no_k7_test[['label_kinetics400', 'label_kinetics600']][~pd.isna(df_no_k7_test['label_kinetics400'])].fillna('unknown')
np_k4_to_k6 = np.hstack((k4_label_enc.encode(df_k4_to_k6.values[:, 0]).reshape(-1,1), k6_label_enc.encode(df_k4_to_k6.values[:, 1]).reshape(-1,1)))
np_k4_to_k6
for (row, col) in np_k4_to_k6:
    k4_to_k6_conf[row, col] += 1
k4_to_k6_conf.sum()
k4_to_k6_conf == k6_to_k7_conf
k4_to_k6_conf == k4_to_k7_conf
np.save('../data/analysis/kinetics/k4_to_k6_confusion', k4_to_k6_conf)
k6_to_k7_conf.argmax(1)
k6_to_k7_conf.argmax(1).shape
k6_to_k7_conf.argmax(1)
k6_to_k7_conf
k6_to_k7_conf[:, 1:]
k6_to_k7_conf[:, 1:].sum(1)
np.where(k6_to_k7_conf[:, 1:].sum(1) == 0)
k6_to_k7_conf[0]
k6_to_k7_conf.shape
k6_label_enc.decode([59, 90])
k6_to_k7_conf.shape
k6_to_k7_conf[90].sum()
k6_to_k7_conf[59].sum()
k6_to_k7_conf[:, 1:].sum(1)
np.where(k6_to_k7_conf[:, 1:].sum(1) < 100)
k6_label_enc.decode([59, 90, 319])
np.where(k6_to_k7_conf[319, 1:] > 0)
np.where(k6_to_k7_conf[319, :] > 0)
k7_label_enc.decode(367)
k7_label_enc.decode([367])
k6_to_k7_conf[319]
k6_to_k7_conf[319].sum()
np.where(k6_to_k7_conf[319, :] > 0)
k6_to_k7_conf[:, 1:].sum(1) < 100
k6_to_k7_conf[:, 1:].sum(1)
k6_to_k7_conf[:, :].sum(1)
k6_to_k7_conf.argmax(1)
np.where(k6_to_k7_conf.argmax(1) == 0)
len(np.where(k6_to_k7_conf.argmax(1) == 0))
np.where(k6_to_k7_conf.argmax(1) == 0).shape
np.where(k6_to_k7_conf.argmax(1) == 0)
np.where(k6_to_k7_conf.argmax(1) == 0)[0]
np.where(k6_to_k7_conf.argmax(1) == 0)[0].shape
unk_argmax_k6_to_k7 = np.where(k6_to_k7_conf.argmax(1) == 0)[0]
unk_argmax_k6_to_k7
k6_label_enc.decode(unk_argmax_k6_to_k7)
unk_argmax_k6_to_k7
k6_to_k7_conf[:, 1:].argmax(1)
k6_to_k7_conf[:, 1:].argmax(1)[unk_argmax_k6_to_k7]
k6_to_k7_conf[:, 1:].argmax(1)[unk_argmax_k6_to_k7] + 1
k6_to_k7_conf[:, 1:].argmax(1)[unk_argmax_k6_to_k7]
k_argmax_select_unks_argmax = k6_to_k7_conf[:, 1:].argmax(1)[unk_argmax_k6_to_k7] + 1
k7_label_enc.decode(k_argmax_select_unks_argmax)
k6_to_k7_conf
k6_label_enc
list(k6_label_enc)
np.array(k6_label_enc)
pd.Series(k6_label_enc)
pd.Series(np.array(k6_label_enc))
pd.Series(np.array(k6_label_enc)).isin(pd.Series(np.array(k7_label_enc)))
pd.Series(np.array(k6_label_enc))[pd.Series(np.array(k6_label_enc)).isin(pd.Series(np.array(k7_label_enc)))]
600 - 597
pd.Series(np.array(k6_label_enc))[~pd.Series(np.array(k6_label_enc)).isin(pd.Series(np.array(k7_label_enc)))]
pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k7_label_enc)))]
len(pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k7_label_enc)))])
pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k6_label_enc)))]
len(pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k6_label_enc)))])
k4_to_k7_conf.argmax(1)
k7_label_enc.decode(k4_to_k7_conf.argmax(1)[54])
list(pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k6_label_enc)))].index)
k4_idx_not_in_k7 = list(pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k6_label_enc)))].index)
k7_label_enc.decode(k4_to_k7_conf.argmax(1)[k4_idx_not_in_k7])
pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k6_label_enc)))]
pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k7_label_enc)))]
k4_series_not_in_k7 = pd.Series(np.array(k4_label_enc))[~pd.Series(np.array(k4_label_enc)).isin(pd.Series(np.array(k7_label_enc)))]
k4_series_not_in_k7
k7_label_enc.decode(k4_to_k7_conf.argmax(1)[k4_series_not_in_k7.index])
k4_to_k7_conf[k4_series_not_in_k7.index]
np.where(k4_to_k7_conf[k4_series_not_in_k7.index] > 0)
k4_to_k7_conf[k4_series_not_in_k7.index, 1:]
k4_to_k7_conf[k4_series_not_in_k7.index, 1:].sum(1)
