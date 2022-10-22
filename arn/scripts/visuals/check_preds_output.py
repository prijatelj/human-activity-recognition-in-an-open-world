"""Checks on the prediction output of different predictors."""
import h5py
import pandas as pd
import numpy as np

from exputils.data import OrderedConfusionMatrices


def check_preds_csvs(prefix, orig, load, verbose=False):
    for split in ['train', 'validate', 'test']:
        print('\nAssessing preds of original == load on split = ', split)
        for i in range(1, 42):
            pre = 'post-feedback' if i%2 else 'new-data'
            i = int(np.floor(i/2))
            orig_train = pd.read_csv(f'{prefix}{orig}/step-{i}_{pre}_predict/{split}/preds.csv')
            load_train = pd.read_csv(f'{prefix}{load}/step-{i}_{pre}_predict/{split}/preds.csv')
            if verbose:
                print(f'{i:2d}, {pre:15s}, {(load_train == orig_train).all()}')
            else:
                print(f'{i:2d}, {pre:15s}, {(load_train == orig_train).all().all()}')


def check_preds_ocms(prefix, orig, load):
    for split in ['train', 'validate', 'test']:
        print('\nAssessing preds of original == load on split = ', split)
        for i in range(1, 42):
            pre = 'post-feedback' if i%2 else 'new-data'
            i = int(np.floor(i/2))

            orig_ocm = OrderedConfusionMatrices.load(f'{prefix}{orig}/step-{i}_{pre}_predict/{split}/preds_top-cm.h5')
            load_ocm = OrderedConfusionMatrices.load(f'{prefix}{load}/step-{i}_{pre}_predict/{split}/preds_top-cm.h5')

            match = orig_ocm.tensor == load_ocm.tensor
            if isinstance(match, bool):
                print(i, match)
            else:
                print(i, match.all())

            print('  labels match = ', (orig_ocm.labels == load_ocm.labels).all())


            orig_nmi = orig_ocm.get_conf_mat().mutual_information('arithmetic')
            load_nmi = load_ocm.get_conf_mat().mutual_information('arithmetic')
            print('  NMI check: ', orig_nmi == load_nmi, orig_nmi, load_nmi)


# To see whose csvs are identical or not
prefix = '/tmp/har/data/results/sim_owr/exp2_2d_sim/test-run/gauss-recog/gmm_finch/'
check_preds_csvs(prefix, 'version_3', 'version_7')
check_preds_ocms(prefix, 'version_3', 'version_7')

# NOTE These above checks do not pass. It is due to floating point error and
# normalization. Consider saving log_probs (unnormalized). This also affects
# the resulting ordered confusion matrices. Perhaps would be fixed by using
# float64.

# To see whose csvs are identical or not for the fine tune ANNs
prefix = '/tmp/har/data/results/sim_owr/exp2_2d_sim/test-run/fine-tune/'
check_preds_csvs(prefix, 'version_8', 'version_16')
check_preds_ocms(prefix, 'version_8', 'version_16')

# Same happens for the fine tune anns on sims...



def check_pre_is_post_preds_ocms(prefix):
    # Check of 0% feedback finetune anns pre == post feedback (cuz it should)
    for split in ['train', 'validate', 'test']:
        print(f'Checking {split} pre == post')
        for i in range(1, 11):
            #pre = 'post-feedback' if i%2 else 'new-data'
            #i = int(np.floor(i/2))
            if split == 'validate' and i >= 6:
                # There is no validate for k700 (validate is used as test)
                break
            pre = OrderedConfusionMatrices.load(f'{prefix}/step-{i}_new-data_predict/{split}/preds_top-cm.h5')
            post = OrderedConfusionMatrices.load(f'{prefix}/step-{i}_post-feedback_predict/{split}/preds_top-cm.h5')
            match = pre.tensor == post.tensor
            if isinstance(match, bool):
                print(i, match)
            else:
                print(i, match.all())

            pre_nmi = pre.get_conf_mat().mutual_information('arithmetic')
            post_nmi = post.get_conf_mat().mutual_information('arithmetic')
            print('NMI check: ', pre_nmi == post_nmi, pre_nmi, post_nmi)

# The above passes? How? The charts indicate a raise in post for everything!
# This would then have to be a visualization error, possibly accidental
# cumulative cms?

prefix = '/tmp/har/data/models/exp2_test-run_5inc-per-dset/kowl_tsf_test-run_skip-fit-1_fine-tune_label_recent_first_5inc-per-dset/version_3/'
check_pre_is_post_preds_ocms(prefix)
