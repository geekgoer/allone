import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/falcon_10ms_hybrid_160k_v1/gap_diagnosis.py"
spec = importlib.util.spec_from_file_location("gap_diagnosis", SCRIPT)
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)


def synthetic():
    rng = np.random.default_rng(10)
    # 10 episodes/environment, 4 correlated rows/episode.
    env = np.repeat(np.arange(10), 40)
    episode = np.repeat(np.arange(100), 4)
    n = len(env)
    return dict(X=rng.normal(size=(n, 12)), U=rng.normal(size=(n, 4)),
                episode_id=episode, environment_id=env, query_group_id=episode,
                source_kind=np.tile(np.arange(4), n // 4),
                control_bucket=np.zeros(n, dtype=np.int8),
                cold_rti_kkt=np.arange(n, dtype=float),
                velocity=rng.normal(size=(n, 3)))


class GapDiagnosticTests(unittest.TestCase):
    def test_exact_stratified_size_and_reproducibility(self):
        labels = np.repeat(np.arange(4), [1, 7, 19, 29])
        a = gap.stratified_rows(labels, 11, np.random.default_rng(4))
        b = gap.stratified_rows(labels, 11, np.random.default_rng(4))
        self.assertEqual(len(a), 11)
        self.assertEqual(len(np.unique(a)), 11)
        np.testing.assert_array_equal(a, b)

    def test_group_isolation_and_identical_environment_validation(self):
        data = synthetic()
        arms, threshold = gap.make_arms(data, 360)
        tr, va = arms["episode_within_environment"]
        self.assertEqual(gap.overlap(data, tr, va)["episode_id"], 0)
        self.assertEqual(gap.overlap(data, tr, va)["query_group_id"], 0)
        self.assertEqual(gap.overlap(data, tr, va)["environment_id"], 10)
        np.testing.assert_array_equal(np.sort(np.r_[tr, va]), np.arange(400))
        for name in ("environment_low_kkt", "environment_matched_random"):
            np.testing.assert_array_equal(arms[name][1], arms["environment"][1])
        low, random = arms["environment_low_kkt"][0], arms["environment_matched_random"][0]
        self.assertEqual(len(low), len(random))
        labels = gap.strata(data)
        for key in np.unique(labels):
            self.assertEqual(int(np.sum(labels[low] == key)), int(np.sum(labels[random] == key)))
        self.assertEqual(threshold, np.quantile(data["cold_rti_kkt"][:360], .95))

    def test_kkt_threshold_does_not_use_validation(self):
        data = synthetic()
        _, before = gap.make_arms(data, 360)
        data["cold_rti_kkt"][360:] = 1e12
        _, after = gap.make_arms(data, 360)
        self.assertEqual(before, after)

    def test_recompute_control_buckets(self):
        u = np.array([[10., 0., 0., 0.], [20., 0., 0., 0.],
                      [10., -3., 0., 0.], [18., 0., 0., 0.]])
        np.testing.assert_array_equal(gap.control_bucket(u), [0, 2, 2, 1])

    def test_prepare_preserves_source_and_does_not_require_test(self):
        data = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src", Path(tmp) / "out"
            src.mkdir()
            gap._atomic_npz(src / "train.npz", gap.take(data, np.arange(360)))
            gap._atomic_npz(src / "validation.npz", gap.take(data, np.arange(360, 400)))
            norm = gap.normalization_from_train(data["X"][:360], data["U"][:360])
            gap.write_json(src / "normalization.json", {k: v.tolist() for k, v in norm.items()})
            gap.write_json(src / "split_manifest.json", {"teacher_policy": "test", "input_contract": "test"})
            before = {p.name: gap.digest(p) for p in src.iterdir()}
            gap.prepare(src, out)
            self.assertEqual(before, {p.name: gap.digest(p) for p in src.iterdir()})
            self.assertFalse((src / "test.npz").exists())
            with self.assertRaises(FileExistsError):
                gap.prepare(src, out)


if __name__ == "__main__":
    unittest.main()
