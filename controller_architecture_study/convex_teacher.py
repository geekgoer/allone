"""Unique acceleration-QP teacher plus an explicit NED/FRD attitude controller.

This is a NEW control baseline, not the original ACADO nonlinear MPC. The QP
optimizes translation; quaternion attitude feedback converts its first action
to collective thrust acceleration and body rates. No hidden previous action.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy
from scipy import sparse

from falcon_paper_fidelity.motion_primitive import _zup_flu_attitude
from state_adapter import normalize_quaternion, quaternion_inverse, quaternion_multiply, quaternion_to_rotation_vector


@dataclass(frozen=True)
class QPConfig:
    horizon: int = 20
    prediction_dt: float = .1
    gravity: float = 9.80665
    position_weight: float = 4.
    velocity_weight: float = 2.
    acceleration_weight: float = .3
    difference_weight: float = .2
    slack_weight: float = 80.
    corridor_radius: tuple = (1.5, 1.5, 1.)
    acceleration_limit: tuple = (6., 6., 5.)
    tilt_degrees: float = 35.
    attitude_gain: float = 3.5
    eps_abs: float = 1e-6
    eps_rel: float = 1e-6
    max_iter: int = 10000


def osqp_module():
    # A scoped wheel keeps the user's existing NumPy/PyTorch installation intact.
    vendor = Path(__file__).parent / "vendor/convex_qp"
    if sys.platform == "win32" and vendor.exists():
        sys.path.insert(0, str(vendor)) if str(vendor) not in sys.path else None
    try:
        import osqp
    except ImportError as exc:
        raise ImportError("Install osqp==1.0.5 using the command in CONVEX_TEACHER_V1.md") from exc
    return osqp


class ConvexTeacher:
    def __init__(self, config=QPConfig()):
        self.settings = config
        self.config = SimpleNamespace(horizon=config.horizon, prediction_dt=config.prediction_dt)
        n, dt = config.horizon, config.prediction_dt
        if (n < 1 or dt <= 0 or min(config.position_weight, config.velocity_weight,
                config.acceleration_weight, config.slack_weight) <= 0 or config.difference_weight < 0
                or not 0 < config.tilt_degrees < 80 or min(config.corridor_radius) <= 0
                or min(config.acceleration_limit) <= 0 or config.acceleration_limit[2] >= config.gravity):
            raise ValueError("invalid strictly convex QP configuration")
        self.position_map = np.zeros((3*n, 3*n))
        self.velocity_map = np.zeros((3*n, 3*n))
        for i in range(n):
            for j in range(i+1):
                self.position_map[3*i:3*i+3, 3*j:3*j+3] = np.eye(3)*dt**2*(i-j+.5)
                self.velocity_map[3*i:3*i+3, 3*j:3*j+3] = np.eye(3)*dt
        identity = np.eye(3*n)
        difference = identity[3:] - identity[:-3]
        h = (config.position_weight*self.position_map.T@self.position_map
             + config.velocity_weight*self.velocity_map.T@self.velocity_map
             + config.acceleration_weight*identity + config.difference_weight*difference.T@difference)
        self.hessian = sparse.block_diag((2*h, 2*config.slack_weight*identity), format="csc")
        self.minimum_eigenvalue_lower_bound = 2*min(config.acceleration_weight, config.slack_weight)
        polygon = np.zeros((8*n, 3*n))
        slope = np.tan(np.deg2rad(config.tilt_degrees))*np.cos(np.pi/8)
        for i in range(n):
            for face in range(8):
                angle = face*np.pi/4
                polygon[8*i+face, 3*i:3*i+3] = [np.cos(angle), np.sin(angle), slope]
        zero = np.zeros_like(identity)
        self.constraints = sparse.csc_matrix(np.vstack((
            np.hstack((identity, zero)), np.hstack((zero, identity)),
            np.hstack((polygon, np.zeros((8*n, 3*n)))),
            np.hstack((self.position_map, -identity)), np.hstack((-self.position_map, -identity)))))
        self.lower = np.r_[-np.tile(config.acceleration_limit, n), np.zeros(3*n), np.full(14*n, -np.inf)]
        self.upper = np.r_[np.tile(config.acceleration_limit, n), np.full(3*n, np.inf),
                           np.full(8*n, slope*config.gravity), np.zeros(6*n)]
        self.solver = osqp_module().OSQP()
        self.solver.setup(P=sparse.triu(self.hessian, format="csc"), q=np.zeros(6*n),
            A=self.constraints, l=self.lower, u=self.upper, verbose=False,
            eps_abs=config.eps_abs, eps_rel=config.eps_rel, max_iter=config.max_iter,
            adaptive_rho=False, rho=.1, polishing=False, warm_starting=True)
        self.reset()

    def signature(self):
        return {"teacher": "strictly_convex_translation_qp_plus_attitude_feedback_v1",
                "osqp_version": osqp_module().__version__, "settings": asdict(self.settings),
                "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                "hessian_eigenvalue_lower_bound": self.minimum_eigenvalue_lower_bound,
                "independent_zero_primal_dual_start": True}

    def reset(self):
        self.records = []

    def query(self, state, references):
        started = time.perf_counter()
        c, n = self.settings, self.settings.horizon
        if len(references) != n+1:
            raise ValueError("QP needs horizon+1 references")
        p = np.concatenate([ref.position-state.position for ref in references[1:]])
        v = np.concatenate([ref.velocity-state.velocity for ref in references[1:]])
        feedforward = np.concatenate([ref.acceleration for ref in references[:-1]])
        free_position = np.concatenate([(i+1)*c.prediction_dt*state.velocity for i in range(n)])
        q = np.r_[2*(c.position_weight*self.position_map.T@(free_position-p)
                     - c.velocity_weight*self.velocity_map.T@v-c.acceleration_weight*feedforward),
                  np.zeros(3*n)]
        upper = self.upper.copy()
        radius = np.tile(c.corridor_radius, n)
        upper[-6*n:-3*n] = radius+p-free_position
        upper[-3*n:] = radius-p+free_position
        if not np.isfinite(q).all() or not np.isfinite(upper[-6*n:]).all():
            raise ValueError("nonfinite QP input")
        self.solver.update(q=q, l=self.lower, u=upper)
        # All label/live queries have the same start, regardless of query order.
        self.solver.warm_start(x=np.zeros(6*n), y=np.zeros(self.constraints.shape[0]))
        result = self.solver.solve(raise_error=False)
        if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
            raise RuntimeError(f"QP did not solve: {result.info.status}")
        evaluated = self.constraints@result.x
        violation = float(max(0., np.max(self.lower-evaluated), np.max(evaluated-upper)))
        stationarity = float(np.max(abs(self.hessian@result.x+q+self.constraints.T@result.y)))
        if violation > 2e-4 or stationarity > 2e-4*max(1., np.max(abs(q))):
            raise RuntimeError(f"QP residual too large: primal={violation}, dual={stationarity}")
        acceleration = result.x[:3]
        thrust = float(np.linalg.norm(acceleration-[0., 0., c.gravity]))
        desired = normalize_quaternion(_zup_flu_attitude(acceleration, references[0].orientation,
                                       c.gravity)*[1., 1., -1., -1.])
        rotation = quaternion_to_rotation_vector(quaternion_multiply(quaternion_inverse(state.quaternion), desired))
        # Planner rates differentiate the PLANNER's thrust attitude, not the QP's
        # optimized attitude. Adding them here would mix incompatible targets.
        rates = np.clip(c.attitude_gain*rotation, [-3., -3., -2.], [3., 3., 2.])
        control = np.r_[thrust, rates]
        if not np.isfinite(control).all() or not 2. <= thrust <= 20.:
            raise RuntimeError("invalid physical allocation")
        return control.astype(np.float32), {"iterations": result.info.iter, "constraint_violation": violation,
            "stationarity": stationarity, "stationarity_scaled": stationarity/max(1., np.max(abs(q))),
            "max_slack": float(max(0., np.max(result.x[3*n:]))),
            "acceleration_bound_active": bool(np.any(abs(result.x[:3*n]) >= np.tile(c.acceleration_limit, n)-1e-4)),
            "tilt_bound_active": bool(np.any(abs(evaluated[6*n:14*n]-upper[6*n:14*n]) <= 1e-4)),
            "first_acceleration": acceleration.copy(), "solve_ms": (time.perf_counter()-started)*1000}

    def solve(self, state, references):
        action, diagnostics = self.query(state, references)
        self.records.append({"U": action, "position": state.position.copy(), "velocity": state.velocity.copy(),
                             "reference_position": references[0].position.copy(), **diagnostics})
        return action
