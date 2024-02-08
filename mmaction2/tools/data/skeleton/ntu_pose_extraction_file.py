from ntu_pose_extraction_copy import file_process
import os
import sys
sys.path.append(r"D:\openmmlab\mmaction2\tools\data\skeleton")

print(os.getcwd())

input_file = (
    r"D:\openmmlab\mmaction2\work_dirs\tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb\badminton_video\val")
output_file = (
    r"D:\openmmlab\mmaction2\work_dirs\tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb\badminton_video\val_output")

for root, dirs, files in os.walk(input_file):
    for file in files:
        print(os.path.join(root, file))
        file_process(os.path.join(root, file), os.path.join(
            output_file, file.split('.')[0]+'.pkl'),)
