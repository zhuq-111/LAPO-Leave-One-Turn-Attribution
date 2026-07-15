# Copyright 2026 LAPO Authors
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

import unittest

import numpy as np

from lapo.algorithm import (
    add_process_advantage,
    build_process_advantages,
    replace_token_span,
    standardize_outcomes,
)


class LapoAlgorithmTest(unittest.TestCase):
    def test_delete_placeholder_preserves_downstream_context(self):
        context = [10, 11, 20, 21, 30, 31]
        self.assertEqual(
            replace_token_span(context, 2, 4, [99]),
            [10, 11, 99, 30, 31],
        )

    def test_group_scaling_spans_trajectories_and_turns(self):
        result = build_process_advantages(
            [[1.0, 2.0], [-1.0], [10.0]],
            ["question-a", "question-a", "question-b"],
            epsilon=1e-6,
        )

        self.assertAlmostEqual(float(result.scales[0]), 1.0 + 1e-6, places=5)
        self.assertAlmostEqual(float(result.scales[1]), 1.0 + 1e-6, places=5)
        self.assertAlmostEqual(float(result.scales[2]), 10.0 + 1e-6, places=5)

    def test_sign_gate_rejects_zero_and_reversed_directions(self):
        result = build_process_advantages(
            [[0.0, 0.1, 10.0]], ["question"], epsilon=1e-6
        )

        # Centering makes the zero-gain turn negative and the smaller positive gain
        # negative. Neither direction is supported by the corresponding raw gain.
        np.testing.assert_array_equal(result.gated[0][:2], np.zeros(2, dtype=np.float32))
        self.assertGreater(float(result.gated[0][2]), 0.0)

    def test_all_zero_gains_add_no_process_advantage(self):
        result = build_process_advantages([[0.0, 0.0]], ["question"], epsilon=1e-6)
        np.testing.assert_array_equal(result.gated[0], np.zeros(2, dtype=np.float32))

    def test_final_and_non_search_turns_are_excluded_from_group_stats(self):
        result = build_process_advantages(
            [[1.0, 0.0, 0.0]],
            ["question"],
            epsilon=1e-6,
            eligible_rows=[[True, False, False]],
        )
        np.testing.assert_array_equal(result.normalized[0], np.zeros(3, dtype=np.float32))

    def test_outcome_advantage_is_retained_when_gate_is_zero(self):
        combined = add_process_advantage(0.75, np.zeros(3, dtype=np.float32), 0.5)
        np.testing.assert_allclose(combined, np.full(3, 0.75, dtype=np.float32))

    def test_outcomes_are_standardized_within_question(self):
        advantages = standardize_outcomes([0.0, 1.0, 1.0], ["a", "a", "b"], 1e-6)
        self.assertLess(float(advantages[0]), 0.0)
        self.assertGreater(float(advantages[1]), 0.0)
        self.assertEqual(float(advantages[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
