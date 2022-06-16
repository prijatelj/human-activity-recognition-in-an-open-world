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
    #log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)
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
logits = []
labels = []
for logit, label in k400_train:
    logits.append(logit)
    labels.append(label)
logits = []
labels = []
missing_logits = []
for i, sample in enumerate(k400_train):
    if sample is None:
        missing_logits.append(i)
    else:
        logits.append(sample[0])
        labels.append(sample[1])
len(logits)
logits[0]
type(logits[0])
t_logits = torch.stack(logits)
t_logits.shape
t_logits.argmax(1)
len(labels)
type(labels[0])
labels[0]
t_labels = torch.stack(labels)
t_labels.shape
t_labels.detach().cpu().numpy()
np_labels = t_labels.detach().cpu().numpy()
np_logits = t_logits.detach().cpu().numpy()
np_labels.shape
np_logits.shape
from exputils.data import OrderedConfusionMatrices
np.hstack((np.zeros(np_logits.shape[0]), np_logits))
np.hstack((np.zeros(np_logits.shape[[0]]), np_logits))
np.zeros(np_logits.shape[0])
np.zeros(np_logits.shape[0]).shape
np.hstack((np.zeros(np_logits.shape[[0]]).reshape(-1,1), np_logits))
np.hstack((np.zeros(np_logits.shape[0]).reshape(-1,1), np_logits))
np.hstack((np.zeros(np_logits.shape[0]).reshape(-1,1), np_logits)).shape
k400_train.label_enc.decode(np.hstack((np.zeros(np_logits.shape[0]).reshape(-1,1), np_logits)), one_hot_axis=-1)
dec_logits = k400_train.label_enc.decode(np.hstack((np.zeros(np_logits.shape[0]).reshape(-1,1), np_logits)), one_hot_axis=-1)
dec_labels = k400_train.label_enc.decode(np_labels, one_hot_axis=-1)
dec_logits
dec_labels
(dec_labels == dec_logits).sum()
(dec_labels == dec_logits).mean()
'unknown' in dec_labels
ocm = OrderedConfusionMatrices(dec_labels, dec_logits, list(k400_train.label_enc), 5)
label_enc = deepcopy(k400_train.label_enc)
label_enc.pop('unknown')
ocm = OrderedConfusionMatrices(dec_labels, dec_logits, list(k400_train.label_enc), 5)
%pdb
ocm = OrderedConfusionMatrices(dec_labels, dec_logits, list(k400_train.label_enc), 5)
ocm = OrderedConfusionMatrices(dec_labels, np_logits, list(label_enc), 5)
dec_labels
dec_labels.reshape(-1,1)
ocm = OrderedConfusionMatrices(dec_labels, np_logits, list(label_enc), 5)
ocm = OrderedConfusionMatrices(dec_labels.reshape(-1,1), np_logits, list(label_enc), 5)
ocm = OrderedConfusionMatrices(label_enc.encode(dec_labels.reshape(-1,1)), np_logits, list(label_enc), 5)
ocm
ocm.accuracy()
ocm.accuracy(5)
ocm.save('../results/k400_start/timesformer/frepr_logits/train_ocm.h5')
ocm.label_enc
list(ocm.label_enc)
