from setuptools import setup, find_packages

setup(
    name="dial-mpc",
    author="Haoru Xue",
    author_email="haoru-xue@berkeley.edu",
    packages=find_packages(
        include=["dial_mpc", "dial_mpc.*", "csm", "csm.*"]
    ),
    version="0.0.2",
    install_requires=[
        "numpy<2.0.0",
        "matplotlib",
        "tqdm",
        "tyro",
        # Pinned to the versions the released models were verified under.
        # MuJoCo was checked at both ends of this range (3.11.0 and 3.14.0).
        "jax[cuda12]==0.6.2",
        "jax-cosmo==0.1.0",
        "mujoco>=3.11,<3.15",
        "mujoco-mjx>=3.11,<3.15",
        "brax==0.14.1",
        "art",
        "emoji",
        "scienceplots",
        "flax==0.10.7",
        "optax==0.2.8",
        "cloudpickle",
        "flask",
        "pillow",
        "setuptools<81",
    ],
    package_data={
        "dial_mpc": ["examples/*.yaml", "models/**/*"],
        "csm": ["*.json"],
    },
    entry_points={
        "console_scripts": [
            "dial-mpc=dial_mpc.core.dial_core:main",
            "dial-mpc-sim2sim=dial_mpc.core.dial_sim2sim:main",
            "dial-mpc-sim2real=dial_mpc.core.dial_sim2real:main",
            "dial-mpc-sim=dial_mpc.deploy.dial_sim:main",
            "dial-mpc-real=dial_mpc.deploy.dial_real:main",
            "dial-mpc-plan=dial_mpc.deploy.dial_plan:main",
            "dial-mpc-weights=dial_mpc.deploy.dial_weights:main",
            "dial-csm=dial_mpc.csm_entrypoint:train",
            "dial-csm-benchmark=dial_mpc.csm_entrypoint:benchmark",
            "dial-csm-eval=dial_mpc.csm_entrypoint:evaluate",
            "dial-rl-prior=dial_mpc.csm_entrypoint:train_rl_prior",
            "dial-residual-mppi=dial_mpc.csm_entrypoint:residual_mppi",
            "dial-score=dial_mpc.csm_entrypoint:dial_score",
            "dial-score-eval=dial_mpc.csm_entrypoint:dial_score_eval",
            "dial-score-serve=dial_mpc.csm_entrypoint:dial_score_serve",
            "dial-live=dial_mpc.csm_entrypoint:dial_live",
            "csm-live=dial_mpc.csm_entrypoint:csm_live",
            "dial-score-compose=dial_mpc.csm_entrypoint:dial_score_compose",
            "dial-score-bank=dial_mpc.csm_entrypoint:dial_score_bank",
        ],
    },
)
