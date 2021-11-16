import pandas as pd
import torch
from finetuning_layers import FinNeTune
from torch.utils.data import TensorDataset, DataLoader

def finetune(features, labels, model, epochs=25, batch_size=1000,verbose=False, save_path=""):
    features_t = features[:int(len(features)*.75)]
    features_v = features[int(len(features)*.75):]
    t_len = len(features_t)
    v_len = len(features_v)
    print(t_len)
    print(v_len)
    labels_t = labels[:int(len(labels)*.75)]
    labels_v = labels[int(len(labels)*.75):]
    features_t = torch.stack(features_t)
    labels_t = torch.stack(labels_t)
    features_v = torch.stack(features_v)
    labels_v = torch.stack(labels_v)
    dataset = TensorDataset(features_t, labels_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_v = TensorDataset(features_v, labels_v)
    dataloader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    model.cuda()
    best_cls_loss = 99999999999999999
    for epoch in range(epochs):
        tot_cls_loss = 0.0
        right = 0
        for i, x in enumerate(dataloader):
            torch.autograd.set_grad_enabled(True)
            sfeatures, slabels = x
            sfeatures= sfeatures.cuda()
            slabels= slabels.cuda()
            garbage, prediction = model(sfeatures)
            right += torch.sum(torch.eq(torch.argmax(prediction, dim=1),torch.argmax(slabels, dim=1)).int()).cpu().numpy()
            loss = criterion(prediction, slabels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_cls_loss += loss.item()
        tacc = str(right/t_len)
        right = 0
        tot_cls_loss = 0.0
        for i, x in enumerate(dataloader_v):
            torch.autograd.set_grad_enabled(False)
            sfeatures, slabels = x
            sfeatures= sfeatures.cuda()
            slabels= slabels.cuda()
            garbage, prediction = model(sfeatures)
            right += torch.sum(torch.eq(torch.argmax(prediction, dim=1),torch.argmax(slabels, dim=1)).int()).cpu().numpy()
            loss = criterion(prediction, slabels)
            tot_cls_loss += loss.item()
        if best_cls_loss > tot_cls_loss:
            print(best_cls_loss)
            best_cls_loss = tot_cls_loss
            torch.save(model.state_dict(),save_path+'finetune_best.pt')
            print("Epoch: " + str(epoch) + "---------------")
            print("New Best " + str(tot_cls_loss))
            print("Train Accuracy: " + tacc)
            print("Val Accuracy: " + str(right / v_len))

def finetune_val(features_t, labels_t, features_v, labels_v, model, epochs=25, batch_size=1000,verbose=False, save_path=""):
    t_len = len(features_t)
    v_len = len(features_v)
    print(t_len)
    print(v_len)
    features_t = torch.stack(features_t)
    labels_t = torch.stack(labels_t)
    features_v = torch.stack(features_v)
    labels_v = torch.stack(labels_v)
    dataset = TensorDataset(features_t, labels_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_v = TensorDataset(features_v, labels_v)
    dataloader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    model.cuda()
    best_cls_loss = 99999999999999999
    for epoch in range(epochs):
        tot_cls_loss = 0.0
        right = 0
        for i, x in enumerate(dataloader):
            torch.autograd.set_grad_enabled(True)
            sfeatures, slabels = x
            sfeatures = sfeatures.cuda()
            slabels = slabels.cuda()
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
        for i, x in enumerate(dataloader_v):
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
            print(best_cls_loss)
            best_cls_loss = tot_cls_loss
            torch.save(model.state_dict(), save_path + 'finetune_best.pt')
            print("Epoch: " + str(epoch) + "---------------")
            print("New Best " + str(tot_cls_loss))
            print("Train Accuracy: " + tacc)
            print("Val Accuracy: " + str(right / v_len))


def load_data(target_path):
    data = pd.read_csv(target_path, index_col=False, header=0)
    data = data.sample(frac=1).reset_index(drop=True)

    features = []
    annotations = []

    for i, row in data.iterrows():
        target = "/media/sgrieggs/scratch365/humongous_big_dumps/par_kinetics/" + row['anonymous_id'].split(".")[
            0] + "_logits.pt"
        features.append(torch.load(target, map_location=torch.device('cpu')))
        # video_names.append(row[1]['anonymous_id'])
        annotations.append(row['class'])
    class_labels_map = {}
    index = 0
    classes = set()
    for row in annotations:
        classes.add(row)
    classes = list(classes)
    classes.sort()
    for x in classes:
        class_labels_map[x] = index
        index += 1
    print(len(class_labels_map))
    one_hots = []
    for x in annotations:
        annotation = torch.zeros(len(class_labels_map))
        annotation[class_labels_map[x]] = 1
        one_hots.append(annotation)
    return features, one_hots


features, one_hots = load_data('/home/sgrieggs/5fold_training_m24/train-0_tr-val/5folds_train-0.csv')
features_v, one_hots_v = load_data('/home/sgrieggs/5fold_training_m24/train-0_tr-val/5folds_val-0.csv')
net = FinNeTune(input_size=400, n_classes=29)
# test = torch.load("/home/sgrieggs/PycharmProjects/arn/arn/finetune_best.pt")
# net.load_interior_weights(test)
# finetune(features,one_hots, net)
finetune_val(features,one_hots, features_v,one_hots_v, net)