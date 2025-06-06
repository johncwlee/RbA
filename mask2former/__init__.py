# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from .data.dataset_mappers.mask_former_semantic_void_dataset_mapper import (
    MaskFormerSemanticVoidDatasetMapper,
)

from .data.dataset_mappers.mask_former_semantic_coco_mix_dataset_mapper import (
    MaskFormerSemanticCocoMixDatasetMapper
)

from .data.dataset_mappers.mask_former_semantic_street_hazards_mapper import (
    MaskFormerSemanticStreetHazardsMapper
)

from .data.dataset_mappers.open_coco_mapper import (
    OpenPanopticCOCODatasetMapper
)

from .data.dataset_mappers.mask_former_semantic_street_hazards_coco_mix_mapper import (
    MaskFormerSemanticStreetHazardsCOCOMixMapper
)

from .data.dataset_mappers.mask_former_semantic_allo_mapper import (
    MaskFormerALLOSemanticDatasetMapper
)
from .data.dataset_mappers.mask_former_semantic_allo_coco_mix_dataset_mapper import (
    MaskFormerALLOCocoMixDatasetMapper
)
from .data.dataset_mappers.mask_former_semantic_coco_mix_dataset_mapper_binary import (
    MaskFormerSemanticCocoMixDatasetMapperBinary
)

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.coco_panoptic_open_evaluator import COCOPanopticOpenEvaluator

