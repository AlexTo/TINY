# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.FCOS_ON = False
_C.MODEL.LOC_ON = False
_C.MODEL.GAU_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""

# add by hui, to ignore some key when load pre-tra
_C.MODEL.IGNORE_WEIGHT_KEYS = []

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True

# ######## add by hui for ScaleResize
_C.INPUT.USE_SCALE = False
_C.INPUT.SCALES = ()
_C.INPUT.SCALE_MODE = 'bilinear'


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

_C.DATASETS.COCO_DATASET = CN()
_C.DATASETS.COCO_DATASET.TRAIN_FILTER_IGNORE = True
# attention: test_filter_ignore only inference dataset, not inference evaluate, cause it will reload gt from file
_C.DATASETS.COCO_DATASET.TEST_FILTER_IGNORE = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
# add by hui, whether use more data augmentation.
_C.DATALOADER.USE_MORE_DA = 0
# add by hui, when use crop DA, how the crop image size
_C.DATALOADER.DA_CROP_SIZE = (None, None)
# add by hui for data augmentation
_C.DATALOADER.DA_WANT_GT_RANGE = (0, None)
_C.DATALOADER.DA_GT_SCALE_RANGE = (0, None)
_C.DATALOADER.DA_MIN_CROP_SIZE_RATIO = 0.5
_C.DATALOADER.DA_MIN_CROP_OVERLAP = ()
_C.DATALOADER.DA_CROP_RESIZE_PROB = 0.5
# add by hui for DA4
_C.DATALOADER.DA4_COLOR_AUG = False
_C.DATALOADER.DA4_SCALE_RANGE = ()        # for multi-scale train (range scale)
_C.DATALOADER.DA4_SCALES = (1.,)             # for multi-scale train (discrete scales)
_C.DATALOADER.DA4_OFFSET_X_RANGE = ()     # for translate transform
_C.DATALOADER.DA4_OFFSET_Y_RANGE = ()
_C.DATALOADER.USE_SCALE_MATCH = False
_C.DATALOADER.SCALE_MATCH = CN()
_C.DATALOADER.SCALE_MATCH.TARGET_ANNO_FILE = ''
_C.DATALOADER.SCALE_MATCH.BINS = 100
_C.DATALOADER.SCALE_MATCH.EXCEPT_RATE = -1.
_C.DATALOADER.SCALE_MATCH.SCALE_MODE = 'bilinear'
_C.DATALOADER.SCALE_MATCH.DEFAULT_SCALE = 1./4
_C.DATALOADER.SCALE_MATCH.SCALE_RANGE = (0., 2.)
_C.DATALOADER.SCALE_MATCH.TYPE = 'ScaleMatch'       # 'MonotonicityScaleMatch', 'ScaleMatch', 'GaussianScaleMatch'
_C.DATALOADER.SCALE_MATCH.USE_LOG_SCALE_BIN = False
_C.DATALOADER.SCALE_MATCH.OUT_SCALE_DEAL = 'clip'    # 'clip', 'use_default_scale'
_C.DATALOADER.SCALE_MATCH.REASPECT = ()
# only for MonotonicityScaleMatch
_C.DATALOADER.SCALE_MATCH.SOURCE_ANNO_FILE = ''
# only for GaussianScaleMatch
_C.DATALOADER.SCALE_MATCH.MU_SIGMA = (0, 1)  # can also use for MonotonicityScaleMatch
_C.DATALOADER.SCALE_MATCH.GAUSSIAN_SAMPLE_FILE = ''
_C.DATALOADER.SCALE_MATCH.USE_MEAN_SIZE_IN_IMAGE = False
_C.DATALOADER.SCALE_MATCH.MIN_SIZE = 0

# add by hui for balance_normal_sampler
_C.DATALOADER.USE_TRAIN_BALANCE_NORMAL = False
_C.DATALOADER.TRAIN_NORMAL_RATIO = 0.5
_C.DATALOADER.USE_TEST_BALANCE_NORMAL = False
_C.DATALOADER.TEST_NORMAL_RATIO = 0.5

_C.DATALOADER.DEBUG = CN()
_C.DATALOADER.DEBUG.CLOSE_SHUFFLE = False

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"
# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
# GN for backbone
_C.MODEL.BACKBONE.USE_GN = False


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False
# add by hui for upsample FPN output feature size
_C.MODEL.FPN.UPSAMPLE_RATE = []
_C.MODEL.FPN.UPSAMPLE_MODE = 'nearest'


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"

_C.MODEL.RPN.NUM_CONVS = 4  # when RPN_HEAD=="MultiConvRPNHead", it will used.


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# add by hui for OHEM
_C.MODEL.RPN.OHEM = 0     # 0 for no ohem used
_C.MODEL.RPN.OHEM1_NEG_RATE = 3.
_C.MODEL.RPN.OHEM2_BATCH_SIZE_PER_IM = _C.MODEL.RPN.BATCH_SIZE_PER_IMAGE
_C.MODEL.RPN.OHEM2_POSITIVE_FRACTION = _C.MODEL.RPN.POSITIVE_FRACTION
_C.MODEL.RPN.OHEM2_HARD_RATE = 1.0
_C.MODEL.ROI_HEADS.OHEM = 0
_C.MODEL.ROI_HEADS.OHEM1_NEG_RATE = 3.
_C.MODEL.ROI_HEADS.OHEM2_BATCH_SIZE_PER_IM = _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
_C.MODEL.ROI_HEADS.OHEM2_POSITIVE_FRACTION = _C.MODEL.ROI_HEADS.POSITIVE_FRACTION
_C.MODEL.ROI_HEADS.OHEM2_HARD_RATE = 1.0

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
# add by hui to FPN select feature map level for paoposal
_C.MODEL.ROI_BOX_HEAD.POOLER_LEVEL_MAP = 'scale'
_C.MODEL.ROI_BOX_HEAD.POOLER_LEVEL_MAP_KWARGS = CN()
_C.MODEL.ROI_BOX_HEAD.POOLER_LEVEL_MAP_KWARGS.LEVEL_MIN = 2   # for 'fixed' level_map
_C.MODEL.ROI_BOX_HEAD.POOLER_LEVEL_MAP_KWARGS.LEVEL_MAX = 5   # for 'fixed' level_map

_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False

_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# add by hui, for remove down sample of resnet for tiny net. (1, 2, 2, 2) for origin
_C.MODEL.RESNETS.RESNET_STAGE_FIRST_STRIDE = tuple()
# add by hui, whether remove pooling in backbone stem.
_C.MODEL.RESNETS.REMOVE_STEM_POOL = False
_C.MODEL.RESNETS.STEM_STRIDE = 2

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000
_C.MODEL.FCOS.CASCADE_ON = False
_C.MODEL.FCOS.CASCADE_AREA_TH = [0.125**2, 0.25 ** 2, 0.5**2, 0.75 ** 2, 1.0]
_C.MODEL.FCOS.CASCADE_NO_CENTERNESS = True
_C.MODEL.FCOS.USE_STRIDE_SCALE_INIT = False
_C.MODEL.FCOS.USE_GN = True
INF = 100000000
_C.MODEL.FCOS.OBJECT_SIZES = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
_C.MODEL.FCOS.DEBUG = CN()
_C.MODEL.FCOS.DEBUG.VIS_LABELS = False

# Focal loss parameter: alpha
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4

# ---------------------------------------------------------------------------- #
# LOC Options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOC = CN()
_C.MODEL.LOC.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.LOC.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.LOC.PRIOR_PROB = 0.01
_C.MODEL.LOC.INFERENCE_TH = 0.05
_C.MODEL.LOC.NMS_TH = 0.6
_C.MODEL.LOC.PRE_NMS_TOP_N = 1000
_C.MODEL.LOC.DEBUG = CN()
_C.MODEL.LOC.DEBUG.VIS_LABELS = False

_C.MODEL.LOC.TARGET_GENERATOR = 'gaussian'       # gausian or fcos
_C.MODEL.LOC.FCOS_CLS_POS_AREA = 1.0            # param for TARGET_GENERATOR == 'fcos'
_C.MODEL.LOC.FCOS_CENTERNESS = False            # param for TARGET_GENERATOR == 'fcos'
_C.MODEL.LOC.FCOS_CENTERNESS_WEIGHT_REG = True  # param for TARGET_GENERATOR == 'fcos'
_C.MODEL.LOC.LABEL_BETA = 2.0                   # for TARGET_GENERATOR in ['gaussian', 'centerness']
_C.MODEL.LOC.CLS_LOSS = 'fixed_focal_loss'      # 'fixed_focal_loss' or 'L2'
_C.MODEL.LOC.CLS_WEIGHT = 1.0
# Focal loss parameter: alpha
_C.MODEL.LOC.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.LOC.LOSS_GAMMA = 2.0
# GHMC loss parameter: momentum
_C.MODEL.LOC.LOSS_GHMC_MOMENTUM = 0.
# GHMC loss parameter: bins
_C.MODEL.LOC.LOSS_GHMC_BINS = 10
_C.MODEL.LOC.LOSS_GHMC_ALPHA = 0.5

# the number of convolutions used in the cls and bbox tower
_C.MODEL.LOC.NUM_CONVS = 4

# add for divide pos.sum() normalization, if not most will use pos.numel() normalization
_C.MODEL.LOC.DIVIDE_POS_SUM = False


# ---------------------------------------------------------------------------- #
# Gaussian Net Options
# ---------------------------------------------------------------------------- #
_C.MODEL.GAU = CN()
_C.MODEL.GAU.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.GAU.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.GAU.PRIOR_PROB = 0.01
_C.MODEL.GAU.INFERENCE_TH = 0.05
_C.MODEL.GAU.NMS_TH = 0.6
_C.MODEL.GAU.PRE_NMS_TOP_N = 1000
_C.MODEL.GAU.DEBUG = CN()
_C.MODEL.GAU.DEBUG.VIS_LABELS = False
_C.MODEL.GAU.DEBUG.VIS_INFER = False

_C.MODEL.GAU.TARGET_GENERATOR = 'gaussian'       # gausian or fcos
_C.MODEL.GAU.LABEL_BETA = 2.0                   # for TARGET_GENERATOR in ['gaussian', 'centerness']
_C.MODEL.GAU.CLS_LOSS = 'fixed_focal_loss'      # 'fixed_focal_loss' or 'L2'
_C.MODEL.GAU.CLS_WEIGHT = 1.0
# Focal loss parameter: alpha
_C.MODEL.GAU.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.GAU.LOSS_GAMMA = 2.0
# GHMC loss parameter: momentum
_C.MODEL.GAU.LOSS_GHMC_MOMENTUM = 0.
# GHMC loss parameter: bins
_C.MODEL.GAU.LOSS_GHMC_BINS = 10
_C.MODEL.GAU.LOSS_GHMC_ALPHA = 0.5

# the number of convolutions used in the cls and bbox tower
_C.MODEL.GAU.NUM_CONVS = 4

_C.MODEL.GAU.C = (0.125, 0.5, 1, 2, 0.75)
_C.MODEL.GAU.LABEL_RADIUS = 1.0

# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4

# ---------------------------------------------------------------------------- #
# FreeAnchor Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.FREEANCHOR = CN()
_C.FREEANCHOR.FREEANCHOR_ON = False
_C.FREEANCHOR.IOU_THRESHOLD = 0.3
_C.FREEANCHOR.PRE_ANCHOR_TOPK = 200
_C.FREEANCHOR.BBOX_REG_WEIGHT = 1.0
_C.FREEANCHOR.BBOX_REG_BETA = 0.11
_C.FREEANCHOR.BBOX_THRESHOLD = 0.5
_C.FREEANCHOR.FOCAL_LOSS_ALPHA = 0.5
_C.FREEANCHOR.FOCAL_LOSS_GAMMA = 2.0

# add by hui from _C.MODEL.xx in old version
_C.FREEANCHOR.USE_GN = False
_C.FREEANCHOR.SPARSE_MASK_ON = False


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""

# add by hui for upsample feature map for detector
_C.MODEL.UPSAMPLE_RATE = []
_C.MODEL.UPSAMPLE_MODE = 'nearest'
_C.MODEL.UPSAMPLE_TRANSFORM_NUM_CONV = 0


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# add by hui, how many iter to test, -1 for never test during train
_C.SOLVER.TEST_ITER = -1
_C.SOLVER.TEST_ITER_RANGE = [1, -1]  # [s, e], e set < 0 means inf
# add by hui, GPU count
_C.SOLVER.NUM_GPU = 8

# add by hui for add new optimizer
_C.SOLVER.OPTIMIZER = 'sgd'
_C.SOLVER.ADAM_BETAS = (0.9, 0.999)
# RPN RCNN head lr factor when anchor change may need bigger one to pretrain.
_C.SOLVER.HEAD_LR_FACTOR = 1.

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100
# add by hui, whether merge result for corner dataset when evaluate MR
_C.TEST.MERGE_RESULTS = False
_C.TEST.MERGE_GT_FILE = ''
# add by hui, for modify coco eval
_C.TEST.USE_IGNORE_ATTR = True  # whether use 'ignore'attr in anno to set it to ignore gt when evaluate
_C.TEST.IGNORE_UNCERTAIN = False
_C.TEST.USE_IOD_FOR_IGNORE = False
_C.TEST.COCO_EVALUATE_STANDARD = 'coco'
# add by hui, for modify voc eval
_C.TEST.VOC_IOU_THS = (0.5,)
_C.TEST.EVALUATE_METHOD = ''  # some bug here. '' for determine by dataset type, 'coco' or 'voc'
# add by hui
_C.TEST.DEBUG = CN()
_C.TEST.DEBUG.USE_LAST_PREDICTION = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

_C.FIXED_SEED = -1  # init seed should >= 0, if < 0 means not use Fixed seed
_C.TEST_FINAL_ITER = True

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False