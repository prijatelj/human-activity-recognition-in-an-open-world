import sys

kuni_csv_path = sys.argv[1]
#   '/media/har//kinetics_unified.csv',
kuni_features_dir = sys.argv[2]
#   '/home/prijatelj/workspace/research/osr/repos/har/data/features/timesformer/timesformer/'
log_file = sys.argv[3]

# Optionally change the suffix of the filenames to load
if len(sys.argv) >= 4:
    ext = sys.argv[4]
    # example: '_logits.pt'
else: # Default
    ext = '_feat.pt'

import logging
logging.basicConfig(
    filename=log_file,
    #filemode='w',
    level=getattr(logging, 'DEBUG', None),
    format='%(asctime)s; %(levelname)s: %(message)s',
    datefmt=None,
)

from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *

kuni = KineticsUnifiedFeatures(
    kuni_csv_path,
    sample_dirs=BatchDirs(
        root_dir=kuni_features_dir,
        batch_col='batch',
    ),
    ext=ext,
    log_warn_file_not_found=True,
)

# This loops through the DataSet, loading all feature reprs if `ext ==
# '_feat.pt'`, or all logits if `ext == '_logits.pt'`. With logging on, the
# log gets filled with DEBUG messages of those that do not exist.
for i in kuni: pass

# Outside of this script, I then parsed that log file with a regex replace in
# vim and saved the UIDs.

# My filenames accidently had a double path separator '/', so I used that with
# the following batch number to remove the filepath before the youtube id:
#   %s/.*\/\/.*\///g
# Then, I removed the ext.
#   %s/_feat.pt//g
# In python these are all equivalent to `line.replace('pattern', '')`


# TODO Perhaps, load the log file (which is live recording right now, so not best
# idea?), and then perform that find replace on contents and save log as new
# file for saving a blacklist of sample uids?
