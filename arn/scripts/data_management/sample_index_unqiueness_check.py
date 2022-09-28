from copy import deepcopy

import yaml

from arn.scripts.visuals.load_results import *
from arn.scripts.visuals.line_plots_inc import *
from copy import deepcopy

import yaml

from arn.scripts.visuals.load_results import *
from arn.scripts.visuals.line_plots_inc import *

kowl = 'arn/scripts/exp2/configs/test-run_kowl_tsf_ftune.yaml'
kowl = docstr_cap(kowl, return_prog=True).environment
kowl = [kowl.start] + kowl.steps

indices = []
for i in range(len(kowl)):
    for split in ['train', 'validate', 'test']:
        split = getattr(kowl[i], split)
        if split is None: continue
        assert all(split.data['sample_index'] == split.data.index)
        indices += list(split.data['sample_index'])

assert len(indices) == np.unique(indices).shape
