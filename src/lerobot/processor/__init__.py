#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import (
    batch_to_transition,
    create_transition,
    transition_to_batch,
)
from .core import (
    EnvAction,
    EnvTransition,
    PolicyAction,
    RobotAction,
    RobotObservation,
    TransitionKey,
)
from .delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from .device_processor import DeviceProcessorStep
from .factory import (
    make_default_processors,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    make_default_teleop_action_processor,
)

# Note: gym_action_processor and hil_processor removed as they are training/simulation related
from .joint_observations_processor import JointVelocityProcessorStep
from .normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    hotswap_stats,
)
from .observation_processor import VanillaObservationProcessorStep
from .pipeline import (
    ActionProcessorStep,
    ComplementaryDataProcessorStep,
    DataProcessorPipeline,
    DoneProcessorStep,
    IdentityProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorKwargs,
    ProcessorStep,
    ProcessorStepRegistry,
    RewardProcessorStep,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    TruncatedProcessorStep,
)
from .policy_robot_bridge import (
    PolicyActionToRobotActionProcessorStep,
    RobotActionToPolicyActionProcessorStep,
)
from .rename_processor import RenameObservationsProcessorStep
from .tokenizer_processor import TokenizerProcessorStep

__all__ = [
    "ActionProcessorStep",
    "ComplementaryDataProcessorStep",
    "batch_to_transition",
    "create_transition",
    "DeviceProcessorStep",
    "DoneProcessorStep",
    "EnvAction",
    "EnvTransition",
    "hotswap_stats",
    "IdentityProcessorStep",
    "InfoProcessorStep",
    "JointVelocityProcessorStep",
    "make_default_processors",
    "make_default_teleop_action_processor",
    "make_default_robot_action_processor",
    "make_default_robot_observation_processor",
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "NormalizerProcessorStep",
    "ObservationProcessorStep",
    "PolicyAction",
    "PolicyActionProcessorStep",
    "PolicyProcessorPipeline",
    "ProcessorKwargs",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RobotAction",
    "RobotActionProcessorStep",
    "RobotObservation",
    "RenameObservationsProcessorStep",
    "RewardProcessorStep",
    "DataProcessorPipeline",
    "AddBatchDimensionProcessorStep",
    "RobotProcessorPipeline",
    "TokenizerProcessorStep",
    "RobotActionToPolicyActionProcessorStep",
    "PolicyActionToRobotActionProcessorStep",
    "transition_to_batch",
    "TransitionKey",
    "TruncatedProcessorStep",
    "UnnormalizerProcessorStep",
    "VanillaObservationProcessorStep",
]
