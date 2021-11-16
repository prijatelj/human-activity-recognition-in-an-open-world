import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FinNeTune(nn.Module):
    def __init__(self, input_size, model_width=512, n_layers=5,out_features=512,n_classes=29,nonlinearity=nn.ReLU):
        self.nonlinearity = nonlinearity
        super().__init__()
        dict = OrderedDict()
        dict["fc0"]=nn.Linear(input_size,model_width)
        dict["nl0"]=nonlinearity()
        for x in range(1,n_layers-1):
            dict['fc'+str(x)] = nn.Linear(model_width,model_width)
            dict['fc' + str(x)] = nonlinearity()
        dict['fc'+str(n_layers-1)] = nn.Linear(model_width,out_features)
        self.fcs = nn.Sequential(dict)
        self.classifier = nn.Linear(out_features, n_classes)

    def forward(self, x):
        x = self.fcs(x)
        classification = F. log_softmax(self.classifier(x),dim=1)
        return x, classification

    def get_finetuned_feature_extractor(self):
        return self.fcs
