"""The ipython script to get the Ordered Confusion Matrix of TimeSformer on
the Kinetics400 train split available.
"""
from arn.models import fine_tune, fine_tune_lit
from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *

k400_train = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    #'/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    sample_dirs=BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/features/timesformer/',
        batch_col='batch_x3d',
    ),
    subset=KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    blacklist='../data/kinetics700_2020/blacklists/timesformer/blacklist-timesformer-kuni-uid.log',
    #k700_suffix_label=False,
    ext='_logits.pt',
    log_warn_file_not_found=True,
    return_label=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)

from exputils.data import OrderedConfusionMatrices

logits = []
labels = []
missing_logits = []
for i, sample in enumerate(k400_train):
    if sample is None:
        missing_logits.append(i)
    else:
        logits.append(sample[0])
        labels.append(sample[1])

t_logits = torch.stack(logits)
t_labels = torch.stack(labels)
np_labels = t_labels.detach().cpu().numpy()
np_logits = t_logits.detach().cpu().numpy()

dec_logits = k400_train.label_enc.decode(np.hstack((np.zeros(np_logits.shape[0]).reshape(-1,1), np_logits)), one_hot_axis=-1)
dec_labels = k400_train.label_enc.decode(np_labels, one_hot_axis=-1)

# Accuracy check
print((dec_labels == dec_logits).mean())

assert 'unknown' not in dec_labels

ocm = OrderedConfusionMatrices(label_enc.encode(dec_labels.reshape(-1,1)), np_logits, list(label_enc), 5)

print('ocm accuracy top 1', ocm.accuracy())
print('ocm accuracy top 5', ocm.accuracy(5))
ocm.save('../results/k400_start/timesformer/frepr_logits/train_ocm.h5')
