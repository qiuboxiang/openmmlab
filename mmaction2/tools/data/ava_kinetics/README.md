# Preparing AVA-Kinetics

## Introduction

<!-- [DATASET] -->

```BibTeX
@article{li2020ava,
  title={The ava-kinetics localized human actions video dataset},
  author={Li, Ang and Thotakuri, Meghana and Ross, David A and Carreira, Jo{\~a}o and Vostrikov, Alexander and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2005.00214},
  year={2020}
}
```

For basic dataset information, please refer to the official [website](https://research.google.com/ava/index.html).
AVA-Kinetics dataset is a crossover between the AVA Actions and Kinetics datasets. You may want to first prepare the AVA datasets. In this file, we provide commands to prepare the Kinetics part and merge the two parts together.

For model training, we will keep reading from raw frames for the AVA part, but read from videos using `decord` for the Kinetics part to accelerate training.

Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ava_kinetics/`.

## Step 1. Prepare the Kinetics700 dataset

The Kinetics part of the AVA-Kinetics dataset are sampled from the Kinetics-700 dataset.

It is best if you have prepared the Kinetics-700 dataset (only videos required) following
[Preparing Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics). We will also have alternative method to prepare these videos if you do not have enough storage (coming soon).

We will need the videos of this dataset (`$MMACTION2/data/kinetics700/videos_train`) and the videos file list (`$MMACTION2/data/kinetics700/kinetics700_train_list_videos.txt`), which is generated by [Step 4 in Preparing Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics#step-4-generate-file-list)

The format of the file list should be:

```
Path_to_video_1 label_1\n
Path_to_video_2 label_2\n
...
Path_to_video_n label_n\n
```

The timestamp (start and end of the video) must be contained. For example:

```
class602/o3lCwWyyc_s_000012_000022.mp4 602\n
```

It means that this video clip is the 12th to 22nd seconds of the original video. It is okay if some videos are missing, and we will ignore them in the next steps.

## Step 2. Download Annotations

Download the annotation tar file (recall that the directory should be located at `$MMACTION2/tools/data/ava_kinetics/`).

```shell
wget https://storage.googleapis.com/deepmind-media/Datasets/ava_kinetics_v1_0.tar.gz
tar xf ava_kinetics_v1_0.tar.gz && rm ava_kinetics_v1_0.tar.gz
```

You should have the `ava_kinetics_v1_0` folder at `$MMACTION2/tools/data/ava_kinetics/`.

## Step 3. Cut Videos

Use `cut_kinetics.py` to find the desired videos from the Kinetics-700 dataset and trim them to contain only annotated clips. Currently we only use the train set of the Kinetics part to improve training. Validation on the Kinetics part will come soon.

Here is the script:

```shell
python3 cut_kinetics.py --avakinetics_anotation=$AVAKINETICS_ANOTATION \
                        --kinetics_list=$KINETICS_LIST \
                        --avakinetics_root=$AVAKINETICS_ROOT \
                        [--num_workers=$NUM_WORKERS ]
```

Arguments:

- `avakinetics_anotation`: the directory to ava-kinetics anotations. Defaults to `./ava_kinetics_v1_0`.
- `kinetics_list`: the path to the videos file list as mentioned in Step 1. If you have prepared the Kinetics700 dataset following `mmaction2`, it should be `$MMACTION2/data/kinetics700/kinetics700_train_list_videos.txt`.
- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `num_workers`: number of workers used to cut videos. Defaults to -1 and use all available cpus.

There should be about 100k videos. It is OK if some videos are missing and we will ignore them in the next steps.

## Step 4. Extract RGB Frames

This step is similar to Step 4 in [Preparing AVA](https://github.com/open-mmlab/mmaction2/tree/main/tools/data/ava#step-4-extract-rgb-and-flow).

Here we provide a script to extract RGB frames using ffmpeg:

```shell
python3 extract_rgb_frames.py --avakinetics_root=$AVAKINETICS_ROOT \
                              [--num_workers=$NUM_WORKERS ]
```

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `num_workers`: number of workers used to extract frames. Defaults to -1 and use all available cpus.

If you have installed denseflow, you can also use `build_rawframes.py` to extract RGB frames:

```shell
python3 ../build_rawframes.py ../../../data/ava_kinetics/videos/ ../../../data/ava_kinetics/rawframes/ --task rgb --level 1 --mixed-ext
```

## Step 5. Prepare Annotations

Use `prepare_annotation.py` to prepare the training annotations. It will generate a `kinetics_train.csv` file containning the spatial-temporal annotations for the Kinetics part, localting at `$AVAKINETICS_ROOT`.

Here is the script:

```shell
python3 prepare_annotation.py --avakinetics_anotation=$AVAKINETICS_ANOTATION \
                              --avakinetics_root=$AVAKINETICS_ROOT \
                              [--num_workers=$NUM_WORKERS]
```

Arguments:

- `avakinetics_anotation`: the directory to ava-kinetics anotations. Defaults to `./ava_kinetics_v1_0`.
- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `num_workers`: number of workers used to prepare annotations. Defaults to -1 and use all available cpus.

## Step 6. Fetch Proposal Files

The pre-computed proposals for AVA dataset are provided by FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks). For the Kinetics part, we use `Cascade R-CNN X-101-64x4d-FPN` from [mmdetection](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth) to fetch the proposals. Here is the script:

```shell
python3 fetch_proposal.py --avakinetics_root=$AVAKINETICS_ROOT \
                          --datalist=$DATALIST \
                          --picklepath=$PICKLEPATH \
                          [--config=$CONFIG ] \
                          [--checkpoint=$CHECKPOINT ]

```

It  will generate a `kinetics_proposal.pkl` file at `$MMACTION2/data/ava_kinetics/`.

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `datalist`: path to the `kinetics_train.csv` file generated at Step 3.
- `picklepath`: path to save the extracted proposal pickle file.
- `config`: the config file for the human detection model. Defaults to `X-101-64x4d-FPN.py`.
- `checkpoint`: the checkpoint for the human detection model. Defaults to the `mmdetection` pretraining checkpoint.

## Step 7. Merge AVA to AVA-Kinetics

Now we are done with the preparations for the Kinetics part. We need to merge the AVA part into the `ava_kinetics` folder (assuming you have AVA dataset ready at `$MMACTION2/data/ava`). First we make a copy of the AVA anotation to the `ava_kinetics` folder (recall that you are at `$MMACTION2/tools/data/ava_kinetics/`):

```shell
cp -r ../../../data/ava/annotations/ ../../../data/ava_kinetics/
```

Next we merge the generated anotation files of the Kinetics part to AVA. Please check: you should have two files `kinetics_train.csv` and `kinetics_proposal.pkl` at `$MMACTION2/data/ava_kinetics/` generated from Step 5 and Step 6. Run the following script to merge these two files into `$MMACTION2/data/ava_kinetics/annotations/ava_train_v2.2.csv` and `$MMACTION2/data/ava_kinetics/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl` respectively.

```shell
python3 merge_annotations.py --avakinetics_root=$AVAKINETICS_ROOT
```

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.

Finally, we need to merge the rawframes of AVA part. You can either copy/move them or generate soft links. The following script is an example to use soft links:

```shell
python3 softlink_ava.py --avakinetics_root=$AVAKINETICS_ROOT \
                        --ava_root=$AVA_ROOT
```

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `ava_root`: the directory to save the ava dataset. Defaults to `$MMACTION2/data/ava`.
