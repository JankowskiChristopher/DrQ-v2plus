import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_agent(cfg: DictConfig) -> None:
    agent_name = cfg.agent_name
    logger.info(f"Running agent {agent_name}")
    if agent_name == "sac":
        from src.agents.SAC.clean_rl_sac import run

        run(cfg)
    elif agent_name == "drqv2":
        from src.agents.drqv2.train import run

        run(cfg)
    else:
        raise NotImplementedError(f"Agent {agent_name} not implemented.")


@hydra.main(config_path="../experiments/configs")
def main(cfg: DictConfig):
    logger.info("Starting experiment with override config:")
    logger.info(OmegaConf.to_yaml(cfg))
    run_agent(cfg)


if __name__ == "__main__":
    main()
