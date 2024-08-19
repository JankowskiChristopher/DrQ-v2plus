from pathlib import Path

from src.agents.drqv2.buffers.replay_buffer import AbstractReplayBuffer, ReplayBufferStorage, make_replay_loader


class DataloaderReplayBuffer(AbstractReplayBuffer):
    def __init__(self, args, data_specs=None):
        assert data_specs is not None
        self.work_dir = Path.cwd()
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            args.buffer_size,
            args.batch_size,
            args.num_workers,
            args.save_snapshot,
            args.nstep,
            args.discount,
        )

        self._replay_iter = None

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def add(self, time_step):
        self.replay_storage.add(time_step)

    def __next__(
        self,
    ):
        return next(self.replay_iter)

    def __len__(
        self,
    ):
        return len(self.replay_storage)
