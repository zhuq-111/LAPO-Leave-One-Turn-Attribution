# Copyright 2026 LOTAPO Authors
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

"""Prompt templates reported in Appendix B of the LOTAPO paper."""


AGENT_PROMPT_TEMPLATE = """You are answering a question that may require external search.

At each step, choose exactly one action:

1. If external information is needed:
<think>Write your brief reasoning about what information is missing and why search is needed.</think>
<search>one concise search query</search>

2. If enough information is available:
<think>Write your brief reasoning leading to the answer.</think>
<answer>final answer only</answer>

Rules:
- Do not output the words "reasoning", "query", or "final answer" as placeholders.
- Use either <search> or <answer>, never both.
- The <search> tag must contain only one concise query.
- Search results may be provided as <information>...</information>; read them but never generate <information>.
- The <answer> tag should contain only the final answer, with no explanation.
- Stop immediately after </search> or </answer>.

Question: {question}"""

GOLD_ANSWER_PREFIX = "<think>Now there's enough information to answer</think>\n<answer>"
GOLD_ANSWER_SUFFIX = "</answer>"


def build_agent_prompt(question: str) -> str:
    """Insert a question into the paper's fixed agent-interaction prompt."""

    return AGENT_PROMPT_TEMPLATE.format(question=question.strip())
