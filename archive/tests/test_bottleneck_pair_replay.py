from itertools import product
from random import Random
import unittest

from controller_architecture_study.bottleneck_pair_replay import minimum_downgrades


class MinimumDowngradesTests(unittest.TestCase):
    def test_matches_exhaustive_minimum_on_ready_batches(self):
        rng = Random(237)
        start, fast, slow = .5, 1.335, 11.788
        for _ in range(120):
            deadlines = [rng.uniform(1., 70.) for _ in range(6)]
            allowed = [rng.choice((True, True, False)) for _ in deadlines]
            order = sorted(range(len(deadlines)), key=deadlines.__getitem__)
            feasible_counts = []
            for choices in product(*[(0, 1) if ok else (0,) for ok in allowed]):
                finish = start
                for i in order:
                    finish += fast if choices[i] else slow
                    if finish > deadlines[i]:
                        break
                else:
                    feasible_counts.append(sum(choices))
            result = minimum_downgrades(deadlines, allowed, start, fast, slow)
            self.assertEqual(None if result is None else sum(result),
                             min(feasible_counts, default=None))

    def test_twelve_jobs_need_nine_eligible_downgrades(self):
        solve = lambda allowed: minimum_downgrades([50.] * 12, allowed, .5, 1.335, 11.788)
        self.assertEqual(sum(solve([True] * 9 + [False] * 3)), 9)
        self.assertIsNone(solve([True] * 8 + [False] * 4))
