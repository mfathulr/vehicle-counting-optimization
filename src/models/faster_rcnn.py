"""Faster R-CNN model with ResNet18 backbone for vehicle detection."""

import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import BasicBlock

from ..config import (
    ANCHOR_ASPECT_RATIOS,
    ANCHOR_SIZES,
    ROI_OUTPUT_SIZE,
    ROI_SAMPLING_RATIO,
)


def create_faster_rcnn_resnet18(num_classes: int) -> FasterRCNN:
    """
    Create a Faster R-CNN model with a modified ResNet18 backbone.

    Args:
        num_classes: Number of classes including background.

    Returns:
        Configured Faster R-CNN model.
    """
    # Build single ResNet18 instance and reuse its components
    resnet = torchvision.models.resnet18(weights=None)

    # Extract layers
    layer1 = resnet.layer1
    layer2 = resnet.layer2
    layer3 = resnet.layer3
    layer4 = resnet.layer4

    # Add extra BasicBlock to each layer for enhanced feature extraction
    layer1.add_module("extra", BasicBlock(64, 64))
    layer2.add_module("extra", BasicBlock(128, 128))
    layer3.add_module("extra", BasicBlock(256, 256))
    layer4.add_module("extra", BasicBlock(512, 512))

    # Assemble backbone
    backbone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        layer1,
        layer2,
        layer3,
        layer4,
    )

    # ResNet18's final layer has 512 output channels
    backbone.out_channels = 512

    # Configure anchor generator for Region Proposal Network
    anchor_generator = AnchorGenerator(
        sizes=ANCHOR_SIZES,
        aspect_ratios=ANCHOR_ASPECT_RATIOS,
    )

    # Configure ROI pooler for feature extraction
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=ROI_OUTPUT_SIZE,
        sampling_ratio=ROI_SAMPLING_RATIO,
    )

    # Build final Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model
