"""AscTec Hummingbird (as used by OmniDrones/NavRL) as an Isaac Lab ``Multirotor``.

Numbers come from OmniDrones' ``hummingbird.yaml`` / ``rotor_group.py``:

* thrust per rotor  = ``KF * throttle**2`` with ``KF = kf * w_max**2 = 8.54858e-6 * 838**2 = 6.003 N``
* rotors on +-x / +-y at 0.17 m, reaction torque = ``-direction * km/kf * thrust`` (km/kf = 0.016)
* rotor lag: ``throttle += 0.43 * (target - throttle)`` per 16 ms step
  => first-order in rotor speed with tau = dt * (1/0.43 - 1) = 0.0212 s (same rpm-domain model as ``Thruster``)
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg

USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hummingbird.usd")

MASS = 0.716  # kg (base 0.68 + 4 x 0.009)
GRAVITY = 9.81
ARM = 0.17
KF = 8.54858e-06  # N / (rad/s)^2
KM = 1.3677728816219314e-07
W_MAX = 838.0  # rad/s
MAX_THRUST = KF * W_MAX**2  # 6.003 N per rotor
HOVER_W = (MASS * GRAVITY / 4.0 / KF) ** 0.5  # ~453 rad/s
DIRECTIONS = (-1.0, 1.0, -1.0, 1.0)
DT = 0.016
TAU = DT * (1.0 / 0.43 - 1.0)

# rows: Fx, Fy, Fz, Tx, Ty, Tz ; columns: rotor_0..3 at angles 0, pi/2, pi, -pi/2 (thrust along +z_b)
ALLOCATION_MATRIX = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, ARM, 0.0, -ARM],  # tau_x =  y * T
    [-ARM, 0.0, ARM, 0.0],  # tau_y = -x * T
    [-d * KM / KF for d in DIRECTIONS],  # tau_z = -direction * km/kf * T
]

HUMMINGBIRD_CFG = MultirotorCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.2,  # OmniDrones RigidBodyPropertiesCfg defaults (robots/config.py)
            angular_damping=0.2,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        rps={".*": HOVER_W},  # "rps" here is rotor speed in rad/s: thrust = thrust_const * rps**2
    ),
    actuators={
        "thrusters": ThrusterCfg(
            thruster_names_expr=["rotor_0", "rotor_1", "rotor_2", "rotor_3"],
            dt=DT,
            thrust_range=(0.0, MAX_THRUST),
            thrust_const_range=(KF, KF),
            tau_inc_range=(TAU, TAU),
            tau_dec_range=(TAU, TAU),
            torque_to_thrust_ratio=KM / KF,
            integration_scheme="euler",
        )
    },
    rotor_directions=list(DIRECTIONS),
    allocation_matrix=ALLOCATION_MATRIX,
)
