import os
import time
from pathlib import Path

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


class Writer:
    """
    Writer class for tensorboard and wandb.
    :ivar args: the arguments from argument parser
    :ivar _writer: the writer
    """

    def __init__(self, args: DictConfig):
        self.args = args
        if not args.run_name:
            self._run_name = f"{args.agent_name}__{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        else:
            self._run_name = args.run_name + f"__{int(time.time())}"  # add time in case of a collision

        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                group=args.wandb_group,
                sync_tensorboard=True,
                config=dict(args),
                name=self._run_name,
                monitor_gym=True,
                save_code=True,
            )
        self._writer = SummaryWriter(f"runs/{self._run_name}", max_queue=100000)
        self._writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @property
    def run_name(self) -> str:
        return self._run_name

    @staticmethod
    def get_entropy_directory() -> str:
        """
        Get the entropy directory depending which one is available on the node.
        Entropy directory differs based on the node.
        :return: the entropy directory
        """
        TRY_DIRS = ["/scidatasm", "/scidatalg"]  # order is important here
        DEFAULT_DIR = f"/home/{os.environ['USER']}"
        for d in TRY_DIRS:
            if Path(d).exists():
                return d
        return DEFAULT_DIR
