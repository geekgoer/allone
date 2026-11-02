from types import SimpleNamespace
import unittest

import numpy as np

from controller_architecture_study.build_context_mode_map import reference_acceleration_bound


class ReferenceEnvelopeTests(unittest.TestCase):
    def test_peak_between_sample_points_is_not_mistaken_for_a_sampled_maximum(self):
        c = np.zeros((3,6))
        c[0,3],c[0,4] = 4./6.,-4./12.  # acceleration4t(1-t)
        window = SimpleNamespace(coefficients=[c],starts=[0.],durations=[1.])
        self.assertGreaterEqual(reference_acceleration_bound(window,0.,1.),1.)
        self.assertAlmostEqual(reference_acceleration_bound(window,0.,.25),.75)

    def test_vector_norm_bounds_every_reference_normal_projection(self):
        c = np.zeros((3,6))
        c[:2,2] = [1.5,2.]
        window = SimpleNamespace(coefficients=[c],starts=[0.],durations=[1.])
        self.assertAlmostEqual(reference_acceleration_bound(window,0.,1.),5.)


if __name__=="__main__":
    unittest.main()
