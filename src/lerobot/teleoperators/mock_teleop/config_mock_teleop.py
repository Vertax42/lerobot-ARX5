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

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("mock_teleop")
@dataclass
class MockTeleopConfig(TeleoperatorConfig):
    n_motors: int = 3
    random_values: bool = True
    static_values: list[float] | None = None
    calibrated: bool = True

    def __post_init__(self):
        if self.n_motors < 1:
            raise ValueError(self.n_motors)

        if self.random_values and self.static_values is not None:
            raise ValueError("Choose either random values or static values")

        if self.static_values is not None and len(self.static_values) != self.n_motors:
            raise ValueError("Specify the same number of static values as motors")
