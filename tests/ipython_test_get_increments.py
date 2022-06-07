# Tests the get_increments() on KineticsUnified labels
import logging
logging.basicConfig(
    #filename='../results/toy_test.log',
    #filemode='w',
    level=getattr(logging, 'DEBUG', None),
    format='%(asctime)s; %(levelname)s: %(message)s',
    datefmt=None,
)
logger = logging.getLogger(__name__)


from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *


k400_train = KineticsUnifiedFeatures(
    '/home/prijatelj/workspace/research/osr/repos/har/data/analysis/kinetics/kinetics_400_600_700_2020.csv',
    '/home/prijatelj/workspace/research/osr/repos/har/data/k700_k400_par_class_map.csv',
    KineticsRootDirs(
        kinetics400_dir='x3d-features-k400/',
        root_dir='/mnt/hdd/workspace/research/osr/har/kitware/'
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
)

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
)

assert k400_train.label_enc == k400_val.label_enc
assert k400_train.label_enc == k400_test.label_enc

dsplits = DataSplits(k400_train, k400_val, k400_test)
known_label_enc = NominalDataEncoder(np.array(k400_train.label_enc)[1:11], unknown_key='unknown')

n_incs = 10
incs = get_increments(
    n_incs,
    dsplits,
    known_label_enc,
    0,
)

logger.debug('Check the type and len of `incs`')
logger.debug('type(incs) = %s', type(incs))
logger.debug('len(incs) = %d', len(incs))

train_samples = 0
val_samples = 0
test_samples = 0

logger.debug('Check the type and len of each increment')
for i, inc in enumerate(incs):
    logger.debug('type(incs[%d]) = %s', i, type(inc))
    logger.debug('len(incs[%d]) = %d', i, len(inc))
    logger.debug('Check the type and len of each split in increment %d', i)

    for j, split in enumerate(inc):
        logger.debug('type(incs[%d][%d]) = %s', i, j, type(split))
        len_split = len(split)
        logger.debug('len(incs[%d][%d]) = %d', i, j, len_split)
        if j == 0:
            train_samples += len_split
        elif j == 1:
            val_samples += len_split
        elif j == 2:
            test_samples += len_split

logger.debug(
    'len(k400_train) = %d; train_samples = %d; == %s',
    len(k400_train),
    train_samples,
    len(k400_train) == train_samples,
)
logger.debug(
    'len(k400_val) = %d; val_samples = %d; == %s',
    len(k400_val),
    val_samples,
    len(k400_val) == val_samples,
)
logger.debug(
    'len(k400_test) = %d; test_samples = %d; == %s',
    len(k400_test),
    test_samples,
    len(k400_test) == test_samples,
)

assert len(k400_train) == train_samples
assert len(k400_val) == val_samples
assert len(k400_test) == test_samples


logger.info(
    'The number of samples across the incremtal splits equals the same '
    'as the source DataSplits. '
    'Now checking the label encoders are the same across all splits in each '
    "increment and the final increment's number of classes is same as source."
)
for i, inc in enumerate(incs):
    inc_label_enc = inc.train.label_enc
    logger.debug('inc %d num labels = %d', i, len(inc_label_enc))
    for j, split in enumerate(inc[1:]):
        assert split.label_enc == inc_label_enc
logger.info('Label Encoders are same for each increment.')
assert len(inc_label_enc) == len(k400_train.label_enc)
logger.info("Final increment's Label Encoders is the same length as source.")
