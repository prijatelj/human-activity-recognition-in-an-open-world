"""Functions that either add args to argparser or post-process those args."""
from arn.data.kinetics_unified import (
    KineticsUnifiedSubset,
    KineticsSplitConfig,
    KineticsRootDirs,
    LabelConfig,
)


def har_dataset_general(parser):
    parser.add_argument(
        'annotation_path',
        help='Filepath to the annotation csv.',
    )
    parser.add_argument(
        'kinetics_class_map_path',
        help='Filepath to the kinetics class mapping csv.',
    )
    parser.add_argument(
        '--bad_samples_dir',
        default=None,
        help='If given, collects and saves DataFrame subset of bad samples',
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Batch size.',
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='If given, DataLoaders shuffle the data.',
    )


def kinetics_root_dirs(parser):
    # KineticsRootDirs
    parser.add_argument(
        'kinetics400_dir',
        help='The video root directory of kinetics400',
    )
    parser.add_argument(
        'kinetics600_dir',
        help='The video root directory of kinetics600',
    )
    parser.add_argument(
        'kinetics700_2020_dir',
        help='The video root directory of kinetics700_2020',
    )


def post_kinetics_root_dirs(args):
    args.kinetic_root_dirs = KineticsRootDirs(
        args.kinetics400_dir,
        args.kinetics600_dir,
        args.kinetics700_2020_dir,
    )

    return args

def single_label_config(parser):
    #parser.add_argument(
    #    'label_config_name',
    #    default=None,
    #    help='Name for the label config if label_confif_known exists.',
    #)
    parser.add_argument(
        '--label_config_yaml',
        default=None,
        help='Filepath to the label config as a YAML file.',
    )

def single_kinetics_unified_subset(parser):
    # Kinetics 400
    parser.add_argument(
        '--kinetics400_train',
        action='store_true',
        help='If given, Kinetics 400 train split is included',
    )
    parser.add_argument(
        '--kinetics400_val',
        action='store_true',
        help='If given, Kinetics 400 validate split is included',
    )
    parser.add_argument(
        '--kinetics400_test',
        action='store_true',
        help='If given, Kinetics 400 test split is included',
    )
    parser.add_argument(
        '--kinetics400_NaN',
        action='store_true',
        help='If given, Kinetics 400 NaN split is included',
    )

    # Kinetics 600
    parser.add_argument(
        '--kinetics600_train',
        action='store_true',
        help='If given, Kinetics 600 train split is included',
    )
    parser.add_argument(
        '--kinetics600_val',
        action='store_true',
        help='If given, Kinetics 600 validate split is included',
    )
    parser.add_argument(
        '--kinetics600_test',
        action='store_true',
        help='If given, Kinetics 600 test split is included',
    )
    parser.add_argument(
        '--kinetics600_NaN',
        action='store_true',
        help='If given, Kinetics 600 NaN split is included',
    )

    # Kinetics 700_2020
    parser.add_argument(
        '--kinetics700_2020_train',
        action='store_true',
        help='If given, Kinetics 700_2020 train split is included',
    )
    parser.add_argument(
        '--kinetics700_2020_val',
        action='store_true',
        help='If given, Kinetics 700_2020 validate split is included',
    )
    parser.add_argument(
        '--kinetics700_2020_test',
        action='store_true',
        help='If given, Kinetics 700_2020 test split is included',
    )
    parser.add_argument(
        '--kinetics700_2020_NaN',
        action='store_true',
        help='If given, Kinetics 700_2020 NaN split is included',
    )


def post_single_kinetics_unified_subset(args):
    # Make KineticsSplitConfigs
    if not (
        args.kinetics400_train
        or args.kinetics400_val
        or args.kinetics400_test
        or args.kinetics400_NaN
        or args.kinetics600_train
        or args.kinetics600_val
        or args.kinetics600_test
        or args.kinetics600_NaN
        or args.kinetics700_2020_train
        or args.kinetics700_2020_val
        or args.kinetics700_2020_test
        or args.kinetics700_2020_NaN
    ):
        # If all are False, assign all 3 to None
        k4_split_conf = None
        k6_split_conf = None
        k7_split_conf = None
    else:
        k4s_split_conf = KineticsSplitConfig(
            args.kinetics400_train,
            args.kinetics400_val,
            args.kinetics400_test,
            args.kinetics400_NaN,
        )
        k6_split_conf = KineticsSplitConfig(
            args.kinetics600_train,
            args.kinetics600_val,
            args.kinetics600_test,
            args.kinetics600_NaN,
        )
        k7_split_conf = KineticsSplitConfig(
            args.kinetics700_2020_train,
            args.kinetics700_2020_val,
            args.kinetics700_2020_test,
            args.kinetics700_2020_NaN,
        )

    if isinstance(args.label_config_yaml, str):
        # TODO load LabelConfig from YAML
        raise NotImplementedError('TODO: Add loading label config from YAML')
    else:
        label_config = None

    # Make KineticsUnifiedSubset
    args.subset =  KineticsUnifiedSubset(
        k4_split_conf,
        k6_split_conf,
        k7_split_conf,
        label_config,
    )

    return args
