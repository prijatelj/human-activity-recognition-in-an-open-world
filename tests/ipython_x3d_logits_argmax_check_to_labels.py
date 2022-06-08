from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *

k400_train = KineticsUnifiedFeatures(
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
    blacklist='../data/kinetics400/blacklists/blacklist-x3d-k4-train-uid.log',
    ext='_logits.pt',
)
k400_val = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/x3d_features/x3d/',
        batch_col='batch_idx',
    ),
    KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(validate=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    blacklist='../data/kinetics400/blacklists/blacklist-x3d-k4-val-uid.log',
    ext='_logits.pt',
)
k400_test = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified_batched.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/x3d_features/x3d/',
        batch_col='batch_idx',
    ),
    KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(test=True),
        labels=LabelConfig('label_kinetics400', known=True)
    ),
    blacklist='../data/kinetics400/blacklists/blacklist-x3d-k4-test-uid.log',
    ext='_logits.pt',
)

logits = np.vstack([k.cpu().numpy() for k in k400_train])
logits_val = np.vstack([k.cpu().numpy() for k in k400_val])
logits_test = np.vstack([k.cpu().numpy() for k in k400_test])

# remove the 'unknown' first label, as X3D only expects 400 classes.
label_enc = deepcopy(k400_train.label_enc)
label_enc.pop('unknown')

(label_enc.decode(logits.argmax(1)) == k400_train.data.labels).mean()

assert k400_val.label_enc == k400_train.label_enc
(label_enc.decode(logits_val.argmax(1)) == k400_val.data.labels).mean()

assert k400_test.label_enc == k400_train.label_enc
(label_enc.decode(logits_test.argmax(1)) == k400_test.data.labels).mean()
