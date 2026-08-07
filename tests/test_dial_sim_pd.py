import unittest
from types import SimpleNamespace

import numpy as np

from dial_mpc.deploy.dial_sim import DialSim


class DialSimPDTest(unittest.TestCase):
    def test_joint_target_pd_uses_current_state_and_clips(self):
        sim = DialSim.__new__(DialSim)
        sim.Nu = 2
        sim.control_kp = np.array(30.0)
        sim.control_kd = np.array(1.0)
        sim.mj_data = SimpleNamespace(
            qpos=np.array([0.0] * 7 + [0.0, 0.0]),
            qvel=np.array([0.0] * 6 + [1.0, -1.0]),
        )
        sim.mj_model = SimpleNamespace(
            actuator_ctrlrange=np.array([[-5.0, 5.0], [-5.0, 5.0]])
        )

        np.testing.assert_allclose(
            sim._joint_target_torque(np.array([0.5, -0.5])),
            [5.0, -5.0],
        )

        # A new measured position must immediately change torque even when the
        # target is unchanged; this is what stale planner-side torque lacked.
        sim.mj_data.qpos[7:] = [0.5, -0.5]
        sim.mj_data.qvel[6:] = 0.0
        np.testing.assert_allclose(
            sim._joint_target_torque(np.array([0.5, -0.5])),
            [0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
