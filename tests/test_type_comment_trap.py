# Copyright 2025 The HuggingFace & XenseRobotics Inc. team. All rights reserved.
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

"""A comment line starting `type:` reads as a PEP 484 type comment.

Several config docstrings show a recipe's typed `gripper:` block, whose first
key is `type`. Written as a plain comment, mypy parses the rest of the line as
a type expression, fails, and — because a syntax error aborts the build — stops
checking the *entire package*. The report points at a comment, so the cause is
not obvious from the message.

Recipe examples in comments therefore carry a `|` gutter. This is the fast
check for that; the mypy pre-commit hook is the slow one.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "lerobot"

# Same shape mypy's lexer looks for: a comment whose content begins with `type:`.
TYPE_COMMENT = re.compile(r"^\s*#\s*type:\s")


def test_no_comment_line_is_mistaken_for_a_type_comment():
    offences = [
        f"{path.relative_to(SRC.parent.parent)}:{n}: {line.strip()}"
        for path in SRC.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if TYPE_COMMENT.match(line)
    ]
    assert not offences, (
        "mypy reads these as PEP 484 type comments, fails to parse them, and stops\n"
        "checking the whole package:\n  " + "\n  ".join(offences) + "\n"
        "In a YAML example, prefix the block with a `|` gutter."
    )
