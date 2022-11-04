import pandas as pd
import torch
from tqdm import tqdm
import sys
from exputils.data import OrderedConfusionMatrices
from exputils.data.labels import NominalDataEncoder
import numpy as np
import os


find_folder = False


kinetics_unified_path = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/" \
                        "kinetics_unified.csv"
kinetics_400_labels = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/" \
                      "arn/arn/data/kinetics400_labels.txt"
kinetics_pred_dir = "/afs/crc.nd.edu/group/cvrl/scratch_31/sgrieggs/dumps/"

sample_list = [["16/-0vzKnJrPQs_000096_000106_logits.pt", "smoking"],
               ["16/-0ML-FXomBw_000001_000011_logits.pt", "playing paintball"],
               ["16/-0IErS_cisg_000017_000027_logits.pt", "marching"]]

# visual_trans_list = ["timesformer", "blurtimesformer", "fliptimesformer",
#                      "inverttimesformer", "jittimesformer",
#                      "noisetimesformer", "rotationtimesformer"]

visual_trans_list = ["x3d", "blur/x3d", "flip/x3d",
                     "invert/x3d", "jit/x3d",
                     "noise/x3d", "rotation/x3d"]


# TODO: Using the base timesformer to find the folders for each sample
if find_folder:
    all_sub_folders = []
    for one_folder in os.listdir(kinetics_pred_dir + visual_trans_list[0]):
        all_sub_folders.append(os.path.join(kinetics_pred_dir + visual_trans_list[0], one_folder))

    target_folders = []
    for one_target_sample in sample_list:
        target_name = one_target_sample[0]

        for one_folder in all_sub_folders:
            if os.path.isfile(os.path.join(one_folder, target_name)):
                target_folders.append(os.path.join(one_folder, target_name))
                print("File found: ", os.path.join(one_folder, target_name))
                break

# TODO: for each sample and each visual transform, load prediction and get result labels
predictions = []

for one_sample in sample_list:
    one_sample_predictions = []
    sample_name = one_sample[0]
    print(sample_name)

    for one_visual in visual_trans_list:
        logits = torch.load(os.path.join(kinetics_pred_dir, one_visual, sample_name),
                            map_location=torch.device('cpu'))
        print(logits.shape)
        pred = torch.argmax(logits).numpy()
        one_sample_predictions.append(int(pred))

    predictions.append(one_sample_predictions)

print(predictions)

# # TODO: Find labels for these samples
# k_unified = pd.read_csv(kinetics_unified_path)
# with open(kinetics_400_labels,'r') as f:
#     k_labels = f.readlines()
#
# print(k_unified)



# if "x3d" in target:
#     model = "x3d"
#     spt = target.split("/")
#     if len(spt) == 7:
#         aug = spt[4]
#     else:
#         aug = "normal"
# elif "timesformer" in target:
#     model = "tf"
#     spt = target.split("/")
#     look = spt[4]
#     if look == 'timesformer':
#         aug = "normal"
#     else:
#         aug = spt[4].split('timesformer')[0]
# else:
#     model = ""
#     aug = ""
#
# print(model)
# print(aug)
#
# k400_2_idx = {}
#
# for x in range(len(labels)):
#     labels[x] = labels[x].strip()
#     k400_2_idx[labels[x]] = x
# label_enc = NominalDataEncoder(
#     labels,
#     unknown_key=None
# )
#
# OrderedConfusionMatrices.load("/mnt/scratch_3/dumps/rotation_fixed/x3d/x3d_rotation_fixed_val_cm.h5")
#
# assert False
#
#
#
# k400train = []
# k400train_cm = []
# k400train_cmp = []
# k400test = []
# k400test_cm = []
# k400test_cmp = []
# k400val_cm = []
# k400val_cmp = []
# k400val = []
# bad = []
#
# for i, row in tqdm(k_uni.iterrows(), total=len(k_uni)):
#     file = row['youtube_id'] +"_"+ str(row['time_start']).zfill(6)+"_"+str(row['time_end']).zfill(6)+"_logits.pt"
#     if row['split_kinetics400'] == 'train':
#         try:
#             test = torch.load(target + "/" + str(row['batch']) + '/' + file)
#         except FileNotFoundError:
#             bad.append(file+'\n')
#             continue
#         except Exception as e:
#             print(e)
#             bad.append(file+'\n')
#             continue
#         pred = torch.argmax(test).detach().cpu().numpy()
#         # print(pred)
#         # print(k400_2_idx[row['label_kinetics400']])
#         k400train.append(pred==k400_2_idx[row['label_kinetics400']])
#         k400train_cm.append(test.detach().cpu().numpy())
#         # ohot = np.zeros(len(label_enc))
#         # ohot[k400_2_idx[row['label_kinetics400']]] = 1
#         k400train_cmp.append(np.array(k400_2_idx[row['label_kinetics400']]))
#     if row['split_kinetics400'] == 'validate':
#         try:
#             test = torch.load(target + "/" + str(row['batch']) + '/' + file)
#         except FileNotFoundError:
#             bad.append(file+'\n')
#             continue
#         except Exception as e:
#             print(e)
#             bad.append(file+'\n')
#             continue
#         pred = torch.argmax(test).detach().cpu().numpy()
#         # print(pred)
#         # print(k400_2_idx[row['label_kinetics400']])
#         k400val_cm.append(test.detach().cpu().numpy())
#         # ohot = np.zeros(len(label_enc))
#         # ohot[k400_2_idx[row['label_kinetics400']]] = 1
#         k400val_cmp.append(np.array(k400_2_idx[row['label_kinetics400']]))
#         k400val.append(pred==k400_2_idx[row['label_kinetics400']])
#     if row['split_kinetics400'] == 'test':
#         try:
#             test = torch.load(target + "/" + str(row['batch']) + '/' + file)
#         except FileNotFoundError:
#             bad.append(file+'\n')
#             continue
#         except Exception as e:
#             print(e)
#             bad.append(file+'\n')
#             continue
#         pred = torch.argmax(test).detach().cpu().numpy()
#         # print(pred)
#         # print(k400_2_idx[row['label_kinetics400']])
#         k400test_cmp.append(np.array(k400_2_idx[row['label_kinetics400']]))
#         k400test_cm.append(test.detach().cpu().numpy())
#         k400test.append(pred==k400_2_idx[row['label_kinetics400']])
#
# ocm_train = OrderedConfusionMatrices(np.expand_dims(np.stack(k400train_cmp),-1),np.stack(k400train_cm), label_enc, top_k=5)
# ocm_train.save(target+'/'+model+"_"+aug+"_train_cm.h5")
# ocm_val = OrderedConfusionMatrices(np.expand_dims(np.stack(k400val_cmp),-1),np.stack(k400val_cm), label_enc, top_k=5)
# ocm_test = OrderedConfusionMatrices(np.expand_dims(np.stack(k400test_cmp),-1),np.stack(k400test_cm), label_enc, top_k=5)
# ocm_val.save(target+'/'+model+"_"+aug+"_val_cm.h5")
# ocm_test.save(target+'/'+model+"_"+aug+"_test_cm.h5")
#
#
# df = pd.DataFrame.from_dict({"k400 train": [sum(k400train)/len(k400train)],"k400 validate":[sum(k400val)/len(k400val)],"k400 test":[sum(k400test)/len(k400test)]})
#
# # df = pd.DataFrame.from_dict({"k400 train": 0,"k400 validate":[sum(k400val)/len(k400val)],"k400 test":[sum(k400test)/len(k400test)]})
#
# df.to_csv(target+'/'+'logit_acc.csv')
#
# with open(target+'/'+'bad_logit2.txt', 'w') as f:
#     f.writelines(bad)