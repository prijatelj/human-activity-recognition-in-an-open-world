import yaml
import numpy as np
import pandas as pd
import sys
import time
from arn.scripts.visuals.load_results import *
from arn.scripts.visuals.measure_novel_react import \
    load_kowl_inc_dsets_with_docstr
import warnings
warnings.filterwarnings("ignore")
import os
import os.path
from tqdm import tqdm

tfs_data_yaml_path = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/" \
                     "arn/arn/scripts/exp2/configs/test-run_kowl_tsf_ftune.yaml"
x3d_data_yaml_path = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/" \
                     "arn/arn/scripts/exp2/configs/test-run_kowl_x3d_ftune.yaml"

visual_trans_base_dir = "/afs/crc.nd.edu/group/cvrl/scratch_31/sgrieggs/dumps/"

model_list = ["x3d", "timesformer"]
vis_trans_list = ["blur", "flip", "invert", "jit", "noise", "rotation"]
phase_list = ["train", "valid", "test"]

tfs_data = load_kowl_inc_dsets_with_docstr(tfs_data_yaml_path)
x3d_data = load_kowl_inc_dsets_with_docstr(x3d_data_yaml_path)



def swap_sample_paths(kinetics_owl_model,
                      visual_transform_data_path,
                      nb_steps=11):
    """Given the loaded Kinetics OWL Experiment and a DataSplits
    object that belongs to the visual transforms.
    You can do this in-place
    """

    # There are 2 models, X3D and TSF
    # There are 6 visual transforms
    # Modify path for visual transform data if model is tsf
    if kinetics_owl_model == "timesformer":
        visual_transform_data_path += "timesformer"

    if kinetics_owl_model == "x3d":
        data = x3d_data
    else:
        data = tfs_data

    # Find all the possible folders for this visual tranform
    all_possible_dir = []
    for path, subdirs, files in os.walk(visual_transform_data_path):
        all_possible_dir.append(path)

    # For each phase (train, valid, test), for each step, swap data path
    for one_phase in phase_list:
        for i in range(nb_steps):
            # Get data according to the phase and step we are at
            if one_phase == "train":
                try:
                    current_step_data = data[i].train.data
                except:
                    current_step_data = None

            elif one_phase == "valid":
                try:
                    current_step_data = data[i].validate.data
                except:
                    current_step_data = None

            elif one_phase == "test":
                try:
                    current_step_data = data[i].test.data
                except:
                    current_step_data = None

            else:
                current_step_data = None
                print("Something is wrong...")

            # For some steps, the validation set is None
            if current_step_data is not None:
                print("Phase: ", one_phase)
                print("Current step: ", i)

                # Directly swap the path for all samples
                # Add columns for youtube id and stuff
                current_step_data["folder_ind"] = current_step_data["sample_path"].str.split('/').str[-2]
                current_step_data["video_name"] = current_step_data["sample_path"].str.split('/').str[-1]
                current_step_data["visual_trans_dir"] = visual_transform_data_path

                # Combine them into a new column with the transform path
                current_step_data['sample_path'] = current_step_data['visual_trans_dir'].astype(str) + '/' + \
                                                   current_step_data['folder_ind'] + '/' + \
                                                   current_step_data['video_name']

            else:
                print("Data is empty." , one_phase, i)



    # TODO: Save the data frame or return? Ask derek


if __name__ == "__main__":
    """
    For each model, each visual tranform:
        each step, process each phase
    """

    for one_model, one_vis in zip(model_list, vis_trans_list):
        print("Process mode: ", one_model)
        swap_sample_paths(kinetics_owl_model=one_model,
                          visual_transform_data_path=visual_trans_base_dir + "/" + one_vis)
