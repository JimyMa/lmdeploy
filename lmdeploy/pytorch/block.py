# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def _div_up(x, n):
    """Perform div up."""
    return (x + n - 1) // n


def _round_up(x, n):
    """Perform round up."""
    return _div_up(x, n) * n


class LogicalTokenBlocks:
    """Logical blocks."""
    ALLOC_SIZE = 128

    def __init__(self, blocks: np.ndarray = None):
        if blocks is None:
            self._blocks = np.zeros((self.ALLOC_SIZE, ), dtype=np.int64)
            self._num_real = 0
        else:
            assert blocks.ndim == 1
            self._blocks = blocks
            self._num_real = len(blocks)
        self.last_shared_node = None

    def reserve(self, size: int):
        """Reserve cache size."""
        num_blocks = self._blocks.size
        if num_blocks >= size:
            return
        reserve_size = _round_up(size - num_blocks, self.ALLOC_SIZE)
        self._blocks = np.pad(self._blocks, (0, reserve_size))

    def __setitem__(self, *args, **kwargs):
        """Set values."""
        return self.get_real_blocks().__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        """Get values."""
        return self.get_real_blocks().__getitem__(*args, **kwargs)

    def get_real_blocks(self):
        """Get logical blocks."""
        return self._blocks[:self._num_real]

    def append(self, blocks: np.ndarray):
        """Append blocks."""
        num_blocks = len(blocks)
        self.reserve(num_blocks + self._num_real)
        slice_start = self._num_real
        slice_end = slice_start + num_blocks
        self._num_real += num_blocks
        self._blocks[slice_start:slice_end] = blocks

    def __len__(self):
        """Get length."""
        return self._num_real

    def resize(self, num_blocks: int):
        """Resize logical blocks."""
        assert num_blocks <= len(self)
        self._num_real = num_blocks

    def reset(self):
        """reset."""
        self.resize(0)
        self.last_shared_node = None

    def clone(self):
        """Clone logical blocks."""
        ret = LogicalTokenBlocks()
        ret.append(self.get_real_blocks())
        return ret
