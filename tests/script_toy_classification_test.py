"""Quick interpreter testing of FineTuneLit."""
import torch


class ToyClassify2D4MVNs(object):
    """Create the generative sampling procedure for obtaining coordinates of
    points as feature data along with the label of which Gaussian distribution
    they belong to. The Gaussian distributions are labeled their index which
    starts at zero at the top most Gaussian centered at [0, 1] and labels the
    rest that follow clockwise around the unit circle.
    """

    def __init__(self, locs=None, scales=0.2, labels=None, seed=None):
        # TODO create PyTorch Gaussian Distributions at locs and scales
        if locs is None:
            locs = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        if not isinstance(scales, list):
            scales = [scales] * len(locs)

        self.mvns = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.Tensor(loc),
                torch.eye(2) * scales[i],
            )
            for i, loc in enumerate(locs)
        ]

    def eq_sample_n(self, num, randperm=True):
        if randperm:
            idx = torch.randperm(num * len(self.mvns))
            return (
                torch.cat([mvn.sample_n(num) for mvn in self.mvns])[idx],
                torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten()[idx],
            )
        return (
            torch.cat([mvn.sample_n(num) for mvn in self.mvns]),
            torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten(),
        )


def setup(seed=0, device="cuda:0"):
    """The setup for all tests in this class."""
    # Set seed: Seems I cannot carry an individual RNG easily...
    torch.manual_seed(seed)

    # Create toy simulation
    toy_sim = ToyClassify2D4MVNs(seed=seed)

    # Create Predictor
    # TODO predictor =

    return toy_sim, predictor


train_num_each = 10
test_num_each = 30
inc_train_num_each = 100
inc_test_num_each = 100
seed = 0
device = "cuda:0"

toy_sim, predictor = setup()

train_features, train_labels = toy_sim.eq_sample_n(train_num_each)



# TODO DataLoader combine the features and labels



fine_tune.fit(train_features, train_labels)



# Generate Test samples
test_features, test_labels = toy_sim.eq_sample_n(test_num_each)

preds = predictor.predict(test_features)

print(
    '(test_features): ',
    (preds.argmax(1) == test_labels).sum().tolist() / len(test_labels),
)

# Generate the train samples
inc_train_features, inc_train_labels = toy_sim.eq_sample_n(inc_train_num_each)

# Append the new train samples to the old samples
train_features = torch.cat([train_features, inc_train_features])
train_labels = torch.cat([train_labels, inc_train_labels])

# Incremental fit by keeping prior points
predictor.fit(train_features, train_labels)

# Generate the incremental test samples
inc_test_features, inc_test_labels = toy_sim.eq_sample_n(
    inc_test_num_each,
)

# Append the new test samples to the old samples
test_features = torch.cat([test_features, inc_test_features])
test_labels = torch.cat([test_labels, inc_test_labels])

preds = predictor.predict(test_features)
print(
    'predictor.predict() first increment and 2nd increment test features: ',
    (preds.argmax(1) == test_labels).sum().tolist() / len(test_labels),
)

"""
# To visualize the results:
import pandas as pd

df = pd.DataFrame(test_features.tolist(),columns=['x','y'])
df['label'] = test_labels.tolist()
df['pred'] = preds.argmax(1).tolist()

import plotly.express as px

fig = px.scatter(
    df,
    x='x',
    y='y',
    color='label',
    symbol='label',
)

fig.show()

fig = px.scatter(
    df,
    x='x',
    y='yi
    color='pred',
    symbol='pred',
)

fig.show()
#"""
