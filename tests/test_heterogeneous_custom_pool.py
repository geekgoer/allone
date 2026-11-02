import numpy as np
from controller_architecture_study.framework_runtime import ControllerPool


def test_additional_existing_plain_controller_keeps_segmented_numerical_equivalence():
    name='plain_mlp_h2_w128_seed0'
    pool=ControllerPool(model_names=[name])
    x=pool.norm['x_mean']+.1*pool.norm['x_std']
    with pool.torch.inference_mode():
        expected=pool.models[name](pool.torch.from_numpy(((x-pool.norm['x_mean'])/pool.norm['x_std'])[None])).numpy()[0]
    expected=expected*pool.norm['u_std']+pool.norm['u_mean']
    np.testing.assert_allclose(pool.predict(name,x),expected,rtol=1e-6,atol=1e-6)
    assert set(pool.provenance()['models'])=={name}
