import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from src.utils.writer import Writer

COMMON_TRAIN_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("buffer_size", "BS", "int"),
    ("fps", "FPS", "float"),
    ("total_time", "T", "time"),
]

COMMON_EVAL_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _format(self, key, value, ty):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.04f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise f"invalid format type: {ty}"

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow") if prefix == "train" else colored(prefix, "green")
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["frame"] = step
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir: Path, use_tb: bool, args: DictConfig):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(log_dir / "train.csv", formating=COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(log_dir / "eval.csv", formating=COMMON_EVAL_FORMAT)

        if use_tb:
            self._sw: Optional[SummaryWriter] = Writer(args).writer
        else:
            self._sw = None

    def log(self, key, value, step):
        if type(value) == torch.Tensor:
            value = value.item()

        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

        mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty: Optional[str]):
        for key, value in metrics.items():
            if ty is not None:
                key = f"{ty}/{key}"
            self.log(f"{key}", value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == "eval":
            self._eval_mg.dump(step, "eval")
        if ty is None or ty == "train":
            self._train_mg.dump(step, "train")

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f"{self._ty}/{key}", value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
