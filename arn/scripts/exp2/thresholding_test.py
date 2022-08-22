"""Testing thresholding optimization."""
from copy import deepcopy
from functools import partial

from scipy.optimize import minimize, minimize_scalar

from exputils.data import ConfusionMatrix

from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *
from arn.models.fine_tune_lit import *

import logging
"""
logging.basicConfig(
    #filename='tmp2.log',
    #filemode='w',
    level=getattr(logging, 'DEBUG', None),
    format='%(asctime)s; %(levelname)s: %(message)s',
    datefmt=None,
)
#"""

k4_val = KineticsUnifiedFeatures(
    '/mnt/hdd/workspace/research/osr/har/kinetics_unified.csv',
    #'/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    sample_dirs=BatchDirs(
        root_dir='/home/prijatelj/workspace/research/osr/repos/har/data/features/x3d/',
        batch_col='batch',
    ),
    subset=KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(validate=True),
        labels=LabelConfig('label_recent_first', known=True)
    ),
    #blacklist='../data/kinetics400/blacklists/blacklist-x3d-k4-train-uid.log',
    #k700_suffix_label=False,
    #ext='_logits.pt',
    log_warn_file_not_found=True,
    #return_index=False,
    #whitelist='../data/kinetics400/first-20_whitelist_test-run.log',
)

targets = k4_val.data.labels.values
preds_argmax = np.ones(len(targets))
preds_max = preds_argmax * 0.8
preds_argmax = k4_val.label_enc.decode(preds_argmax.astype(int))

# Ensure there are some correct preditions
preds_argmax[1000:] = targets[1000:]

# Use label enc to copy
label_enc = deepcopy(k4_val.label_enc)

# Get a subset of target classes to be changed to unknown as truth.
unique_target_subset = np.unique(targets[:100])

unique_pred_subset = set(
    unique_target_subset[:np.floor(len(unique_target_subset) / 2).astype(int)]
)
unique_target_subset = set(unique_target_subset) - unique_pred_subset

# Specify a different level of certainty for a subset of target unknown clases
# Set the threshold to increase for every different class in unique pred subset
threshes = np.linspace(0, 0.8, len(unique_pred_subset))
for i, x in enumerate(unique_pred_subset):
    preds_max[targets == x] = threshes[i]
    label_enc.pop(x)

# Mod targets to change a set of classes within to 'unknown', st unknown exists
for x in unique_target_subset:
    label_enc.pop(x)

mod_targets = label_enc.decode(label_enc.encode(targets))
unique_mod_targets = np.unique(mod_targets, return_counts=True)

measure_func = lambda x: -(x.mcc() ** 2)
result = minimize_scalar(
    #partial(unk_thresh_opt, measure_func=measure_func, is_key=True),
    partial(unk_thresh_opt, is_key=True),
    #x0=[0.0],
    args=(
        mod_targets,
        preds_argmax,
        preds_max,
        k4_val.label_enc,
    ),
    bounds=[0.0, 1.0],
    options={'maxiter': 50},
    method='bounded',
)

# Manually check the ConfusionMatrix
cm = ConfusionMatrix(mod_targets, preds_argmax, labels=k4_val.label_enc)
reduced_cm = cm.reduce(['unknown'], 'known', inverse=True)

# Loop thru thresh to check unk thresh opt with the given measure_func
measures = []
for thresh in threshes:
    measures.append(unk_thresh_opt(
        thresh,
        mod_targets,
        preds_argmax,
        preds_max,
        k4_val.label_enc,
        measure_func=measure_func,
    ))
