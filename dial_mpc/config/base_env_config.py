from dataclasses import dataclass


@dataclass
class BaseEnvConfig:
    task_name: str = "default"
    randomize_tasks: bool = False  # Whether to randomize the task.
    # P gain, or a list of P gains for each joint.
    kp: float = 30.0
    # D gain, or a list of D gains for each joint.
    kd: float = 1.0
    debug: bool = False
    # dt of the environment step, not the underlying simulator step.
    dt: float = 0.02
    # timestep of the underlying simulator step. user is responsible for making sure it matches their model.
    timestep: float = 0.02
    backend: str = "mjx"  # backend of the environment.
    # control method for the joints, either "torque" or "position"
    leg_control: str = "torque"
    action_scale: float = 1.0  # scale of the action space.
    # Recompute the joint PD torque at every physics substep instead of holding
    # the torque computed at the start of the control step.  A real quadruped
    # runs its joint PD on the motor driver far faster than the planner rate;
    # holding one torque across the whole control step is a simulator artifact.
    # Only has an effect when dt > timestep (n_frames > 1); with a single
    # substep the two paths are identical by construction.
    pd_substep: bool = False
    # End an episode when a joint leaves its range.  MuJoCo enforces joint
    # limits with a soft constraint that does not fully converge in one step, so
    # this fires on a sub-degree solver residual while the robot is walking
    # normally -- it measures constraint slack, not a fall.  Set it False to
    # terminate only on the two conditions that mean the robot is actually down
    # (tipped over, body too low).  It does not enter the reward, so changing it
    # alters bookkeeping only, never the dynamics.
    terminate_on_joint_limit: bool = True
