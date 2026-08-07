# DIAL-MPC: Diffusion-Inspired Annealing For Legged MPC

<div align="center">

ICRA 2025, Best Paper Finalist

[[Website]](https://lecar-lab.github.io/dial-mpc/)
[[PDF]](https://drive.google.com/file/d/1Z39MCvnl-Tdraon4xAj37iQYLsUh5UOV/view?usp=sharing)
[[Arxiv]](https://arxiv.org/abs/2409.15610)

[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<img src="assets/joint.gif" width="600px"/>

</div>

This repository contains the code (simulation and real-world experiments with minimum setup) for the paper "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing".

DIAL-MPC is a sampling-based MPC framework for legged robot ***full-order torque-level*** control with both precision and agility in a ***training-free*** manner. 
DIAL-MPC is designed to be simple and flexible, with minimal requirements for specific reward design and dynamics model. It directly samples and rolls out in physics-based simulations (``Brax``) and does not require reduced-order modeling, linearization, convexification, or predefined contact sequences.
That means you can test out the controller in a plug-and-play manner with minimum setup.

## News

- 05/19/2025: 🫰 New demo for ball-spinning on finger can be run with `dial-mpc --example allegro_reorient`.
- 04/24/2025: 🎉 DIAL-MPC made into the best paper final list of ICRA 2025.
- 11/03/2024: 🎉 Sim2Real pipeline is ready! Check out the [Sim2Real](#deploy-in-real-unitree-go2) section for more details.
- 09/25/2024: 🎉 DIAL-MPC is released with open-source codes! Sim2Real pipeline coming soon!

https://github.com/user-attachments/assets/f2e5f26d-69ac-4478-872e-26943821a218


## Table of Contents

1. [Install](#install-dial-mpc)
2. [Synchronous Simulation](#synchronous-simulation)
3. [Asynchronous Simulation](#asynchronous-simulation)
4. [Deploy in Real](#deploy-in-real-unitree-go2)
5. [Writing Your Own Environment](#writing-custom-environment)
6. [Rendering Rollouts](#rendering-rollouts-in-blender)
7. [Citing this Work](#bibtex)

## Simulation Setup

### Install `dial-mpc`

> [!IMPORTANT]
> We recommend Ubuntu >= 20.04 + Python >= 3.10 + CUDA >= 12.3.
> You can create a mamba (or conda) environment before proceeding.

Our environment is Ubuntu 22.04 + Python 3.10 + CUDA 12.6.

```bash
git clone https://github.com/LeCar-Lab/dial-mpc.git --depth 1
cd dial-mpc
pip3 install -e .
```

## Synchronous Simulation

In this mode, the simulation will wait for DIAL-MPC to finish computing before stepping. It is ideal for debugging and doing tasks that are currently not real-time.

#### Run Examples

List available examples:

```bash
dial-mpc --list-examples
```

Run an example:

```bash
dial-mpc --example unitree_h1_jog
```

Run DIAL-TC-MPPI (DIAL annealing with TC-MPPI's conditional,
time-correlated sampling distribution):

```bash
dial-mpc --example unitree_go2_trot_tc
```

The controller is selected by `time_correlated: true`.  Its main tuning
parameters are `tc_history_length`, the `d + 1` entries in
`tc_derivative_weights`, `tc_mean_mode`, and `tc_importance_scale`.  The
stable default `tc_mean_mode: dial` preserves DIAL's annealed mean while
using TC-MPPI's temporal covariance.  The original DIAL-MPC
path remains the default.  See
[`docs/dial_tc_mppi_ko.md`](docs/dial_tc_mppi_ko.md) for the equations and
implementation mapping.

After rollout completes, go to `127.0.0.1:5000` to visualize the rollouts.

## Compositional Energy Policy

The bundled `csm` package trains one scalar trajectory-energy head for each
raw objective (`tracking`, `stability`, `gait`).  At inference it forms
`E_omega(o,U) = omega @ E(o,U)` and performs bounded gradient descent on the
action nodes.  This preserves zero-shot weight composition at the energy
level; finite-sample MPPI updates are deliberately not linearly combined.
The package is JAX/Brax-native and does not require Hydrax, GPC, Warp, or
Torch.

For zero-shot preference composition, the recommended controller is the
anchor-factorized Gibbs score model.  It uses DIAL-TC-MPPI's correlated
candidate proposal, learns the clean candidate logits of only the configured
full-rank anchor weights, composes unseen weights analytically before the
softmax, and executes the resulting learned MPPI update without physics
rollouts or action-gradient optimization:

```bash
dial-afgs --example unitree_go2_trot_tc --smoke
```

Run the deploy-scale pipeline with the same three anchor modes in
`csm/go2_modes.json`:

```bash
dial-afgs --example unitree_go2_trot_tc \
  --samples 2048 --model-candidates 64 --collect-steps 400 \
  --train-iters 30000 --dagger-rounds 3 \
  --dagger-steps 200 --dagger-train-iters 15000 \
  --batch-size 128
```

AFGS stores grouped candidate sets and anchor logits in `gibbs_data.npz`.
No weights from the interior of the simplex are sampled during training.  An
unseen preference `omega` is converted to anchor coordinates by solving
`W.T @ alpha = omega`; candidate logits are combined as `L @ alpha` and only
then passed through the Gibbs softmax.  A shared per-query logit scale is
essential: normalizing rewards separately for each anchor would destroy this
algebraic composition.

Run the end-to-end Go2 integration check:

```bash
dial-csm --example unitree_go2_trot --smoke
```

Run the deploy-scale collection, energy training, three query-level DAgger
rounds, and rollout checkpoint selection:

```bash
dial-csm --example unitree_go2_trot \
  --samples 2048 --collect-steps 400 \
  --energy-candidates 64 --teacher-repeats 8 \
  --train-iters 30000 --dagger-rounds 3 \
  --dagger-steps 200 --dagger-train-iters 15000 \
  --batch-size 256 --calibration-weight 0.1 --sobolev-weight 0.1 \
  --energy-steps 8 --energy-step-size 1.0 --trust-radius 0.05 \
  --minimum-mean-vx 0.5
```

Each run writes `energy_data.npz` (and the compatibility alias `scores.npz`),
`policy.pkl`, `checkpoint_selection.json`, and `visualization.html` below
`csm_runs/`.  These generated files are ignored by Git.

Deploy-scale runs persist data and checkpoints while they are running:

- `energy_data.npz` contains raw objective costs and exact-DIAL direction
  anchors.
- `dagger_round_N.npz` and `dagger_aggregated.npz` persist on-policy data.
- `checkpoints/step_XXXXXX/policy.pkl` and `metadata.json` are saved every
  5,000 gradient steps by default.
- `run_config.json` records energy normalization and the selected rollout.

Base data and each DAgger round are sampled with equal probability.  Objective
normalization is recomputed as an equal-mixture statistic at each round; the
energy heads are exactly reparameterized so their raw energy predictions do
not jump when those statistics change.  Checkpoint survival credit is granted
only to rollouts whose mean forward speed reaches `--minimum-mean-vx`.
Every newly collected guidance query also stores the exact Brax rollout
Jacobian `d[tracking,stability,gait]/dU`; `--sobolev-weight` matches those
objective gradients to the corresponding energy heads.  Older NPZ datasets
remain loadable but their missing Jacobians are masked out.

### Tune Go2 expert weights live

The asynchronous Go2 planner reads the three reward weights
`tracking, stability, gait` from shared memory every MPC cycle.  On a
headless server, start these three processes in separate terminals:

The order matters: the simulator creates the shared-memory channels, so wait
until terminal 1 prints its viewer address before starting terminals 2 and 3.

```bash
# Terminal 1: MuJoCo simulation and browser video stream
MUJOCO_GL=egl dial-mpc-sim --example unitree_go2_trot_deploy \
  --web-viewer-port 8081

# Terminal 2: DIAL-MPC expert planner
dial-mpc-plan --example unitree_go2_trot_deploy

# Terminal 3: live sliders and candidate-matrix tools
dial-mpc-weights --serve --port 8080
```

Forward both ports from a local machine and open `http://127.0.0.1:8080`:

```bash
ssh -N \
  -L 8080:127.0.0.1:8080 \
  -L 8081:127.0.0.1:8081 \
  USER@SERVER
```

For a live DIAL-TC-MPPI simulation, use the corresponding deploy example:

```bash
# Terminal 1: simulator and browser stream
MUJOCO_GL=egl dial-mpc-sim --example unitree_go2_trot_tc_deploy \
  --web-viewer-port 8081

# Terminal 2: DIAL-TC-MPPI planner
dial-mpc-plan --example unitree_go2_trot_tc_deploy
```

Forward `8081` and open `http://127.0.0.1:8081` locally:

```bash
ssh -N -L 8081:127.0.0.1:8081 USER@SERVER
```

The control page embeds the live simulation, applies slider changes on the
next planning cycle, and can save up to three candidate vectors.  It reports
their matrix rank and condition number and stores them in
`csm_runs/go2_weight_candidates.json`.

Weights can also be inspected or changed without the browser:

```bash
dial-mpc-weights --get
dial-mpc-weights --set 2,2,1
```

If the robot falls, use the `Reset robot` button or reset it from another
terminal.  This restores the MuJoCo home keyframe and clears the planner's
previous action plan together:

```bash
dial-mpc-weights --reset
```

Use the saved expert modes to cover the state distribution during energy-data
collection.  The energy heads themselves correspond to raw objectives, not
these three modes:

```bash
dial-csm --example unitree_go2_trot \
  --mode-weights csm/go2_modes.json \
  --samples 2048 --collect-steps 400 \
  --energy-candidates 64 --teacher-repeats 8 \
  --train-iters 30000 --batch-size 256
```

The collector calls the real DIAL `MBDPI.reverse_once`, including its reward
standardization, temperature, clipping, and spline conversion.  Its bounded
update is used only as an energy-gradient direction anchor; raw unweighted
rollout costs supervise the scalar objective heads.

Compare the three standard full-rank expert modes under identical seeds and
initial conditions.  This writes JSON/CSV metrics and one rollout HTML per
mode:

```bash
dial-csm-benchmark --example unitree_go2_trot \
  --weights "2,2,1;2,1,2;1,2,2" --steps 300 --warmup-steps 50
```

Evaluate a saved CSM policy continuously with a real-time browser viewer:

```bash
dial-csm-eval \
  --policy csm_runs/unitree_go2_walk-20260804-084109/policy.pkl \
  --example unitree_go2_trot --omega 2,2,1 \
  --steps 500 --episodes 0 --web-viewer-port 8082
```

Forward `8082` over SSH and open `http://127.0.0.1:8082` locally.

## Asynchronous Simulation

The asynchronous simulation is meant to test the algorithm before Sim2Real. The simulation rolls out in real-time (or scaled by `real_time_factor`). DIAL-MPC will encounter delay in this case.

When DIAL-MPC cannot finish the compute in the time defined by `dt`, it will spit out warning. Slight overtime is accepetable as DIAL-MPC maintains a buffer of the previous step's solution and will play out the planned action sequence until the buffer runs out.

List available examples:

```bash
dial-mpc-sim --list-examples
```

Run an example:

In terminal 1, run

```bash
dial-mpc-sim --example unitree_go2_seq_jump_deploy
```
This will open a mujoco visualization window.

In terminal 2, run

```bash
dial-mpc-plan --example unitree_go2_seq_jump_deploy
```


## Deploy in Real (Unitree Go2)

### Overview

The real-world deployment procedure is very similar to asynchronous simulation.

We use `unitree_sdk2_python` to communicate with the robot directly via CycloneDDS.

### Step 1: State Estimation

For state estimation, this proof-of-concept work requires external localization module to get base **position** and **velocity**.

The following plugins are built-in:

- ROS2 odometry message
- Vicon motion capture system

#### Option 1: ROS2 odometry message

Configure `odom_topic` in the YAML file. You are responsible for publishing this message at at least 50 Hz and ideally over 100 Hz. We provide an odometry publisher for Vicon motion capture system in [`vicon_interface`](https://github.com/LeCAR-Lab/vicon_interface).

> [!CAUTION]
> All velocities in ROS2 odometry message **must** be in **body frame** of the base to conform to [ROS odometry message definition](https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html), although in the end they are converted to world frame in DIAL-MPC.

#### Option 2: Vicon (no ROS2 required)

1. `pip install pyvicon-datastream`
2. Change `localization_plugin` to `vicon_shm_plugin` in the YAML file.
3. Configure `vicon_tracker_ip`, `vicon_object_name`, and `vicon_z_offset` in the YAML file.

#### Option 3: Bring Your Own Plugin

We provide a simple ABI for custom localization modules, and you need to implement this in a python file in your workspace, should you consider not using the built-in plugins.

```python
import numpy as np
import time
from dial_mpc.deploy.localization import register_plugin
from dial_mpc.deploy.localization.base_plugin import BaseLocalizationPlugin

class MyPlugin(BaseLocalizationPlugin):
    def __init__(self, config):
        pass

    def get_state(self):
        qpos = np.zeros(7)
        qvel = np.zeros(6)
        return np.concatenate([qpos, qvel])

    def get_last_update_time(self):
        return time.time()

register_plugin('custom_plugin', plugin_cls=MyPlugin)
```

> [!CAUTION]
> When writing custom localization plugin, velocities should be reported in **world frame**.

> [!NOTE]
> Angular velocity source is onboard IMU. You could leave `qvel[3:6]` in the returned state as zero for now.

Localization plugin can be changed in the configuration file. A `--plugin` argument can be supplied to `dial-mpc-real` to import a custom localization plugin in the current workspace.

### Step 2: Installing `unitree_sdk2_python`

> [!NOTE]
> If you are already using ROS2 with Cyclone DDS according to [ROS2 documentation on Cyclone DDS](https://docs.ros.org/en/humble/Installation/DDS-Implementations/Working-with-Eclipse-CycloneDDS.html), you don't have to install Cyclone DDS as suggested by `unitree_sdk2_python`. But do follow the rest of the instructions.

Follow the instructions in [`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python).

### Step 3: Configuring DIAL-MPC

In `dial_mpc/examples/unitree_go2_trot_deploy.yaml` or `dial_mpc/examples/unitree_go2_seq_jump.yaml`, modify `network_interface` to match the name of the network interface connected to Go2.

Alternatively, you can also pass `--network_interface` to `dial-mpc-real` when launching the robot, which will override the config.

### Step 4: Starting the Robot

Follow the [official Unitree documentation](https://support.unitree.com/home/en/developer/Quick_start) to disable sports mode on Go2. Lay the robot flat on the ground like shown.

<div style="text-align: center;">
    <img src="images/go2.png" alt="Unitree Go2 laying flat on the ground." style="width:50%;">
</div>

### Step 5: Running the Robot

List available examples:

```bash
dial-mpc-real --list-examples
```

Run an example:

In terminal 1, run

```bash
# source /opt/ros/<ros-distro>/setup.bash # if using ROS2
dial-mpc-real --example unitree_go2_seq_jump_deploy
```

This will open a mujoco visualization window. The robot will slowly stand up. If the robot is squatting, manually lift the robot into a standing position. Verify that the robot states match the real world and are updating.

You can supply additional arguments to `dial-mpc-real`:

- `--custom-env`: custom environment definition.
- `--network-interface`: override network interface configuration.
- `--plugin`: custom localization plugin.

Next, in terminal 2, run

```bash
dial-mpc-plan --example unitree_go2_seq_jump_deploy
```

## Writing Custom Environment

1. If custom robot model is needed, Store it in `dial_mpc/models/my_model/my_model.xml`.
2. Import the base environment and config.
3. Implement required functions.
4. Register environment.
5. Configure config file.

Example environment file (`my_env.py`):

```python
from dataclasses import dataclass

from brax import envs as brax_envs
from brax.envs.base import State

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
import dial_mpc.envs as dial_envs

@dataclass
class MyEnvConfig(BaseEnvConfig):
    arg1: 1.0
    arg2: "test"

class MyEnv(BaseEnv):
    def __init__(self, config: MyEnvConfig):
        super().__init__(config)
        # custom initializations below...

    def make_system(self, config: MyEnvConfig) -> System:
        model_path = ("my_model/my_model.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        # TODO: implement reset

    def step(self, state: State, action: jax.Array) -> State:
        # TODO: implement step

brax_envs.register_environment("my_env_name", MyEnv)
dial_envs.register_config("my_env_name", MyEnvConfig)
```

Example configuration file (`my_env.yaml`):
```yaml
# DIAL-MPC
seed: 0
output_dir: dial_mpc_ws/my_model
n_steps: 400

env_name: my_env_name
Nsample: 2048
Hsample: 25
Hnode: 5
Ndiffuse: 4
Ndiffuse_init: 10
temp_sample: 0.05
horizon_diffuse_factor: 1.0
traj_diffuse_factor: 0.5
update_method: mppi


# Base environment
dt: 0.02
timestep: 0.02
leg_control: torque
action_scale: 1.0

# My Env
arg1: 2.0
arg2: "test_2"
```

Run the following command to use the custom environment in synchronous simulation. Make sure that `my_env.py` is in the same directory from which the command is run.

```bash
dial-mpc --config my_env.yaml --custom-env my_env
```

You can also run asynchronous simulation with the custom environment:

```bash
# Terminal 1
dial-mpc-sim --config my_env.yaml --custom-env my_env

# Terminal 2
dial-mpc-plan --config my_env.yaml --custom-env my_env
```

## Rendering Rollouts in Blender

If you want better visualization, you can check out the `render` branch for the Blender visualization examples. 

## Acknowledgements

* This codebase's environment and RL implementation is built on top of [Brax](https://github.com/google/brax).
* We use [Mujoco MJX](https://github.com/deepmind/mujoco) for the physics engine.
* Controller design and implementation is inspired by [Model-based Diffusion](https://github.com/LeCAR-Lab/model-based-diffusion).


## BibTeX

If you find this code useful for your research, please consider citing:

```bibtex
@misc{xue2024fullordersamplingbasedmpctorquelevel,
      title={Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing}, 
      author={Haoru Xue and Chaoyi Pan and Zeji Yi and Guannan Qu and Guanya Shi},
      year={2024},
      eprint={2409.15610},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.15610}, 
}
```
