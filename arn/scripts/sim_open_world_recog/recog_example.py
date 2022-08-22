"""Example script of using novelty recognition Gaussian Recognizer.

After installing this package within the frozen environment/container, generate
the simulation data using:
    `python arn/scripts/sim_open_world_recog/sim_gen [dir_to_hold_data]`
Then you may load in the data using `sim_dir` and the following script.
"""

sim_dir = '/mnt/scratch_3/sim_data/'

from arn.data.kinetics_unified import *
from arn.data.kinetics_owl import *

# Load train data
sim_k4_train = KineticsUnifiedFeatures(
    f'{sim_dir}sim_kunified.csv',
    sample_dirs=BatchDirs(
        root_dir=sim_dir,
        batch_col='batch',
    ),
    subset=KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kunified', known=True)
    ),
    return_label=True,
    one_hot=False,
)

# Load test data
sim_k4_test = KineticsUnifiedFeatures(
    f'{sim_dir}sim_kunified.csv',
    sample_dirs=BatchDirs(
        root_dir=sim_dir,
        batch_col='batch',
    ),
    subset=KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(test=True),
        labels=LabelConfig('label_kunified', known=True)
    ),
    return_label=True,
    one_hot=False,
)

sim_k6_train = KineticsUnifiedFeatures(
    f'{sim_dir}sim_kunified.csv',
    sample_dirs=BatchDirs(
        root_dir=sim_dir,
        batch_col='batch',
    ),
    subset=KineticsUnifiedSubset(
        kinetics600=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kunified', known=True)
    ),
    return_label=True,
    one_hot=False,
)

# """Examples of the other sims:
sim_k6_train = KineticsUnifiedFeatures(
    f'{sim_dir}sim_kunified.csv',
    sample_dirs=BatchDirs(
        root_dir=sim_dir,
        batch_col='batch',
    ),
    subset=KineticsUnifiedSubset(
        kinetics600=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kunified', known=True)
    ),
    return_label=True,
    one_hot=False,
)
sim_k7_train = KineticsUnifiedFeatures(
    f'{sim_dir}sim_kunified.csv',
    sample_dirs=BatchDirs(
        root_dir=sim_dir,
        batch_col='batch',
    ),
    subset=KineticsUnifiedSubset(
        kinetics700_2020=KineticsSplitConfig(train=True),
        labels=LabelConfig('label_kunified', known=True)
    ),
    return_label=True,
    one_hot=False,
)

#
# """Example of loading ALL train data including all splits.
# # Note this totals 126000 samples. 42K each data split.
sim_all = KineticsUnifiedFeatures(
    f'{sim_dir}sim_kunified.csv',
    sample_dirs=BatchDirs(
        root_dir=sim_dir,
        batch_col='batch',
    ),
    subset=KineticsUnifiedSubset(
        kinetics400=KineticsSplitConfig(
            train=True,
            validate=True,
            test=True,
        ),
        kinetics600=KineticsSplitConfig(
            train=True,
            validate=True,
            test=True,
        ),
        kinetics700_2020=KineticsSplitConfig(
            train=True,
            validate=True,
            test=True,
        ),
        labels=LabelConfig('label_kunified', known=True)
    ),
    return_label=True,
    one_hot=False,
)


# Obtain features and labels as tensors
features = []
labels = []
for x, y in sim_k4_train:
    features.append(x)
    labels.append(y)
features = torch.stack(features)
labels = torch.stack(labels)

test_features = []
test_labels = []
for x, y in sim_all:
    test_features.append(x)
    test_labels.append(y)
test_features = torch.stack(test_features)
test_labels = torch.stack(test_labels)

# Visualize the feature space w/ labels
def visualize_feature_repr(df, features, labels='label_kunified', opacity=0.3):
    import plotly.express as px
    fig = px.scatter(
        df,
        x=features[:, 0],
        y=features[:, 1],
        color=labels,
        opacity=opacity,
    )
    fig.show()

# Init and fit the Gaussian Recognizer
from arn.models.novelty_recog.gaussian import GaussianRecog
recog = GaussianRecog(10, 0.1)
recog.fit(features, labels, sim_k4_train.label_enc)

# Obtain the unknowns (outliers) given the recognizer



test_labels_masked = []
for x in range(len(test_features)):
    px = test_features[x][0]
    py = test_features[x][1]
    one = px>=-py
    two = px>=py
    if one and not two:
        test_labels_masked.append(1)
    elif one and two:
        test_labels_masked.append(2)
    elif not one and two:
        test_labels_masked.append(3)
    elif not one and not two:
        test_labels_masked.append(4)
    else:
        assert False

test_labels_masked = torch.tensor(test_labels_masked)

#test_recogs = recog.predict(test_features, test_labels_masked, sim_k4_train.label_enc)
test_recogs = recog.detect(test_features, test_labels_masked, sim_k4_train.label_enc)
recog_unknown_mask = test_recogs == sim_k4_train.label_enc.unknown_idx
unknown_labels = test_recogs[recog_unknown_mask]
unknown_features = test_features[recog_unknown_mask]

# Visualize Outliers detected within the test set, which any detection of
# unknown is actually incorrect since this is "closed set", all classes seen in
# this case. So this is the margin of error, and can be used to set threshold
# with the validation set.
# visualize_feature_repr(sim_k4_test, test_features, test_recogs

# TODO perform recognition with recog.recognize()
import matplotlib.pyplot as plt
plt.scatter(test_features[:, 0], test_features[:, 1], c=test_labels)
plt.show()
plt.scatter(test_features[:, 0], test_features[:, 1], c=test_labels_masked)
plt.show()
plt.scatter(unknown_features[:, 0], unknown_features[:, 1])
plt.show()
recog_preds, recog_label_enc = recog.recognize(
    unknown_features,
    unknown_labels,
    sim_k4_train.label_enc
)
