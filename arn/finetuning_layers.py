from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

class FinNeTune(nn.Module):
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
        classification = F. log_softmax(self.classifier(x), dim=1)
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
