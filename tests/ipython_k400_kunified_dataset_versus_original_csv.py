from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *
k400 = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/x3d_features/x3d/',
        batch_col='batch_idx',
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
import pandas as pd
k4train = pd.read_csv('../data/kinetics400/train.csv', header=True)
k4train = pd.read_csv('../data/kinetics400/train.csv')
k4train
(k4train.split == 'train').all()
k4train.split == 'train'
k400
len(k400)
len(k4train)
get_filename(k4train)
get_filename(k4train, ext='_feat.pt')
k400.data
k400.data.columns
k400.data['filename'] = get_filename(k400.data, ext='_feat.pt')
k400.data['filename']
k4train
k4train['filename'] = get_filename(k4train, ext='_feat.pt')
k4train['filename']
k4train['youtube_id'] == k400.data['youtube_id']
(k4train['youtube_id'] == k400.data['youtube_id']).all()
(k4train['filename'] == k400.data['filename']).all()
(k4train['label'] == k400.data['label']).all()
(k4train['label'] == k400.data['labels']).all()
k4train = pd.read_csv('../data/kinetics400/validate.csv')
k4train = pd.read_csv('../data/kinetics400/train.csv')
k4val = pd.read_csv('../data/kinetics400/validate.csv')
k4test = pd.read_csv('../data/kinetics400/test.csv')
k4test
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
k400
k400_val.data['filename'] = get_filename(k400_val.data, ext='_feat.pt')
k400_val.data['filename']
k4val['filename'] = get_filename(k4val, ext='_feat.pt')
k4val['filename']
k400_val
k400_val.data
k400_val.data.filename == k4val.filename
k400_val.data.filename
k4val.filename
k400_val.data.filename.values == k4val.filename.values
(k400_val.data.filename.values == k4val.filename.values).all()
(k400_val.data.labels.values == k4val.labels.values).all()
(k400_val.data.labels.values == k4val.label.values).all()
k400_test.data['filename'] = get_filename(k400_test.data, ext='_feat.pt')
k400_test = KineticsUnifiedFeatures(
    '/home/prijatelj/workspace/research/osr/repos/har/data/analysis/kinetics/kinetics_400_600_700_2020.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    KineticsRootDirs(
        kinetics400_dir='x3d-features-k400/',
        root_dir='/mnt/hdd/workspace/research/osr/har/kitware/'
    ),
    KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(test=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    #blacklist='../data/kinetics400/x3d-k400-blacklist-train-file.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    #log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
k400_test.data['filename'] = get_filename(k400_test.data, ext='_feat.pt')
k4test['filename'] = get_filename(k4test, ext='_feat.pt')
(k400_test.data.filename.values == k4test.filename.values).all()
len(k400_test.data)
len(k4test.filename)
(k400_test.data.labels.values == k4test.labels.values).all()
(k400_test.data.labels.values == k4test.label.values).all()

# All these euqal True... meaning not only are they the same filename to label,
# they are even the same order, cuz i didn't misordered values at all above.
