"""Fine tuning models. These follow fine tuning in Pre-Trained Models"""
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO Fine Tuning state: init, load, save
# TODO Train/fit code given feature repr input
# TODO extract/predict code given feature repr input


class FineTune(object):
    """The Fine Tuning module of the open world human activity predictor.

    Attributes
    ----------
    model : torch.nn.Module
        The model to be used for fine tuning. This is expected to support
        FineTuneFCANN.
    fit_args : dict()
        The hyperparameters for fitting the fine tuning model. This includes
        the optimizer, epochs, etc. to run a complete run of fitting the model.

        Though, batch size is handled by DataLoaders I believe. Let dataloaders
        handle what they can and their outputs be intput args. Everything will
        be torch.Tensors in and torch.Tensors out.
    device : torch.device()
        the device on which model should be trained
        default: cpu
    eval_args : dict() = None
        The hyperparameters for evaluating the fine tuning model, if any.
        Often, there are none.

    [add anything else you deem necessary as state for the FineTune object.]
    """
    def __init__(
        self,
        model,
        fit_args=None,
        device=torch.device("cpu"),
        eval_args=None,
    ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                'Expected model typed as `torch.nn.Module`, not {type(model)}'
            )
        self.model = model
        self.device = device
        if fit_args is None:
            self.batch_size = 1000
            self.epochs = 25
        else:
            self.batch_size = fit_args['batch_size']
            self.epochs = fit_args['epochs']

    def fit(self, features_t, labels_t, features_v, labels_v,  verbose=False):
        """Fits the model with fit_args and the given features and labels in a
        supervised learning fashion.
        features_t labels_t: features and labels that the model should be trained on.
        features_v labels_v: features and labels that the model should be validated on.
        """
        t_len = len(features_t)
        v_len = len(features_v)
        print(t_len)
        print(v_len)
        # features_t = torch.stack(features_t)
        # labels_t = torch.stack(labels_t)
        # features_v = torch.stack(features_v)
        # labels_v = torch.stack(labels_v)

        dataset = torch.utils.data.TensorDataset(features_t, labels_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataset_val = torch.utils.data.TensorDataset(features_v, labels_v)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=True)
        model = self.model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        best_cls_loss = 99999999999999999
        best_val_acc = 0
        for epoch in range(self.epochs):
            if verbose:
                print("Epoch: " + str(epoch) + "---------------")
            tot_cls_loss = 0.0
            right = 0
            for i, x in enumerate(dataloader):
                torch.autograd.set_grad_enabled(True)
                sfeatures, slabels = x
                sfeatures = sfeatures.to(self.device)
                slabels = slabels.to(self.device)
                garbage, prediction = model(sfeatures)
                right += torch.sum(
                    torch.eq(torch.argmax(prediction, dim=1), torch.argmax(slabels, dim=1)).int()).cpu().numpy()
                loss = criterion(prediction, slabels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tot_cls_loss += loss.item()
            tacc = str(right / t_len)
            right = 0
            tot_cls_loss = 0.0
            for i, x in enumerate(dataloader_val):
                torch.autograd.set_grad_enabled(False)
                sfeatures, slabels = x
                sfeatures = sfeatures.cuda()
                slabels = slabels.cuda()
                garbage, prediction = model(sfeatures)
                right += torch.sum(
                    torch.eq(torch.argmax(prediction, dim=1), torch.argmax(slabels, dim=1)).int()).cpu().numpy()
                loss = criterion(prediction, slabels)
                tot_cls_loss += loss.item()
            if best_cls_loss > tot_cls_loss:
                best_cls_loss = tot_cls_loss
                # torch.save(model.state_dict(), save_path + 'finetune_best.pt')
                best_model = copy.deepcopy(model).cpu()
                if verbose:
                    print("New Best " + str(tot_cls_loss))
                    print("Train Accuracy: " + tacc)
                    print("Val Accuracy: " + str(right / v_len))
        self.model = best_model
        # return

    def extract(self, features):
        """Given features, outputs the fully connected encoding of them.
        Args
        ----
        features : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.model.fcs(features)

    def predict(self, features, labels):
        # TODO If this eval/fwd pass loop overlaps with training loop, reuse
        # the code by function call, otherwise just do eval loop here. more
        # likely the training loop would use this at least in validation.

        # NOTE for our paper, we want this with ability to find a threshold
        # from all train and val data.
        features, prediction = self.model(features)
        return prediction

    def save(self, filepath):
        # this should work
        # args).

        torch.save(self.model, filepath)

        # TODO save args and other state info

    @staticmethod
    def load(
        filepath,
        input_size=400,
        model_width=512,
        n_layers=5,
        out_features=512,
        n_classes=29,
        nonlinearity=nn.ReLU,
        **kwargs,
    ):
        """Load all the pieces from a file(s), possibly be done thru a config
        file. For now, feel free to hard code expectation of FineTuneFCANN.

        Returns
        -------
        FineTune
            An instance of FineTune is returned given the files containing
            state information.
        """

        state_dict = torch.load(filepath) # handles torch model when saved as .pt
        model = FineTuneFCANN(input_size,model_width,n_layers,out_features,n_classes,nonlinearity)
        model.load(state_dict)
        # TODO load the other parameters . . .
        #   If not loaded from file or whatever, then can simply be given by
        #   **kwargs.

        return FineTune(model, **kwargs)


class FineTuneFCANN(nn.Module):
    """Fully Connected Dense ANN for fine tuning."""
    def __init__(
        self,
        input_size,
        model_width=512,
        n_layers=5,
        out_features=512,
        n_classes=29,
        nonlinearity=nn.ReLU,
    ):
        """Fine-tuning ANN consisting of fully-connected dense layers."""
        self.nonlinearity = nonlinearity
        super().__init__()
        ord_dict = OrderedDict()
        ord_dict["fc0"] = nn.Linear(input_size, model_width)
        ord_dict["nl0"] = nonlinearity()

        for x in range(1,n_layers-1):
            ord_dict[f'fc{x}'] = nn.Linear(model_width, model_width)
            ord_dict[f'fc{x}'] = nonlinearity()

        ord_dict[f'fc{n_layers - 1}'] = nn.Linear(model_width, out_features)
        self.fcs = nn.Sequential(ord_dict)
        self.classifier = nn.Linear(out_features, n_classes)

    def forward(self, x):
        """Returns the last fully connected layer and the probs classifier"""
        x = self.fcs(x)
        classification = F.log_softmax(self.classifier(x), dim=1)
        return x, classification

    def get_finetuned_feature_extractor(self):
        """Returns ANN that outputs the final fully connected layer."""
        return self.fcs

    def load_interior_weights(self, state_dict):
        temp = []
        for x in state_dict:
            if x[:3] != 'fcs':
                temp.append(x)
        for x in temp:
            state_dict.pop(x)
        self.load_state_dict(state_dict,strict=False)
