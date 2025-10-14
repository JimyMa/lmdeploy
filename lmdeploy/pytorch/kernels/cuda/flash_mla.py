# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

torch.set_printoptions(threshold=float('inf'))

def flash_mla_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, num_heads_q, head_dim_v).
    """

    # logger.error(f"call flash_mla.py flash_mla_fwd")

    # ===== 打印入参（带 RANK 信息） =====
    # try:
    #     rank = torch.distributed.get_rank()
    # except (RuntimeError, ValueError):
    #     # 未初始化分布式或单卡情况下
    #     rank = 0

    # print(f"[RANK{rank}] [flash_mla_fwd] block_table:\n{block_table}", flush=True)
    # print(f"[RANK{rank}] [flash_mla_fwd] cache_seqlens:\n{cache_seqlens}", flush=True)

    # import inspect
    # frame = inspect.currentframe()
    # args, _, _, values = inspect.getargvalues(frame)
    # for name in args:
    #     val = values[name]
    #     prefix = f"[RANK{rank}] [flash_mla_fwd] {name}:"
    #     if isinstance(val, torch.Tensor):
    #         print(f"{prefix} Tensor, dtype={val.dtype}, shape={tuple(val.shape)}",flush=True)
    #     else:
    #         print(f"{prefix} {val}",flush=True)
    # ====================================

    import flash_mla

    # print(f"q.shape: {q.shape}, k_cache.shape: {k_cache.shape}, block_table.shape: {block_table.shape}, cache_seqlens: {cache_seqlens}, head_dim_v: {head_dim_v}")

    out, lse = flash_mla.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
    )

    # ===== 打印返回值（带 RANK 信息） =====
    # print(f"[RANK{rank}] [flash_mla_fwd] return out: Tensor, dtype={out.dtype}, shape={tuple(out.shape)}",flush=True)
    # print(f"[RANK{rank}] [flash_mla_fwd] return lse: Tensor, dtype={lse.dtype}, shape={tuple(lse.shape)}",flush=True)
    # =====================================

    return out.squeeze(1)
