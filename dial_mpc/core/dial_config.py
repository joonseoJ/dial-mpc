from dataclasses import dataclass


@dataclass
class DialConfig:
    # exp
    seed: int = 0
    output_dir: str = "output"
    n_steps: int = 100
    # env
    env_name: str = "unitree_h1_walk"
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 16  # horizon of samples
    Hnode: int = 4  # node number for control
    Ndiffuse: int = 2  # number of diffusion steps
    Ndiffuse_init: int = 10  # number of diffusion steps for initial diffusion
    temp_sample: float = 0.06  # temperature for sampling
    # factor to scale the sigma of horizon diffuse
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5  # factor to scale the sigma of trajectory diffuse
    update_method: str = "mppi"  # update method
    sigma_scale: float = 1.0  # factor to scale the sigma of control
    # DIAL-TC-MPPI: replace iid node noise with the conditional,
    # time-correlated Gaussian of TC-MPPI.  Disabled for backward compatibility.
    time_correlated: bool = False
    tc_history_length: int = 2
    # "dial" keeps DIAL's annealed proposal mean and applies TC covariance.
    # "conditional" enables the paper's conditional prior/sampling means.
    tc_mean_mode: str = "dial"
    # Weights R^(0)..R^(d) in eq. (15).  Higher derivatives already include
    # powers of 1 / node_dt, so their weights should normally be small.
    tc_derivative_weights: tuple[float, ...] = (1.0, 0.0, 3e-5)
    tc_importance_scale: float = 1.0
