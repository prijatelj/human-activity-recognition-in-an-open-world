"""Script for splitting the given data CSV into multiple data CSVs for each
train and test set per fold.
"""
import csv
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import exputils.io


def parse_args(parser):
    parser.add_argument(
        'labels_filepath',
        help='The filepath to the PAR label CSV.',
    )
    parser.add_argument(
        'output_dir',
        default=None,
        help='The directory filepath used to save the resulting TSVs.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='The seed used to initialize the Kfold CV split.',
    )
    parser.add_argument(
        '-d',
        '--delimiter',
        default=',',
        help='The delimiter used by the PAR labels CSV.',
    )
    parser.add_argument(
        '--stratified_label_set',
        default='ontology_id',
        help='The label set id (column) to stratifiy about for the KFold CV.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If true, creates a new directory if one already exists.',
    )


if __name__ == '__main__':
    args = exputils.io.parse_args(
        ['kfold'],
        parse_args,
        description='Splits the given labels TSV into multiple',
    )

    # Load the given data tsv
    labels = pd.read_csv(
        args.labels_filepath,
        sep=args.delimiter,
        quoting=csv.QUOTE_NONE,
    )

    # Data index splitting: Use kfold (stratified if told to) to split the data
    if args.kfold_cv.stratified:
        # Split the data into stratified folds, preserving percentage of
        # samples for each class among the folds.
        fold_indices = StratifiedKFold(
            args.kfold_cv.kfolds,
            shuffle=args.kfold_cv.shuffle,
            random_state=args.seed,
        ).split(labels.index, labels[args.stratified_label_set])
    else:
        fold_indices = KFold(
            args.kfold_cv.kfolds,
            shuffle=args.kfold_cv.shuffle,
            random_state=args.seed,
        ).split(labels.index)

    # Ensure the output directory exists
    output_dir = exputils.io.create_dirs(args.output_dir)

    # For each fold, save the train and test data TSVs.
    for i, (train_fold, test_fold) in enumerate(fold_indices):
        labels.iloc[train_fold].to_csv(
            os.path.join(
                output_dir,
                f'{args.kfold_cv.kfolds}folds_train-{i}.csv',
            ),
            sep=args.delimiter,
            index=False,
            quoting=csv.QUOTE_NONE,
        )

        labels.iloc[test_fold].to_csv(
            os.path.join(
                output_dir,
                f'{args.kfold_cv.kfolds}folds_test-{i}.csv',
            ),
            sep=args.delimiter,
            index=False,
            quoting=csv.QUOTE_NONE,
        )
