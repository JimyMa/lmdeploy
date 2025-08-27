from typing import Optional, Tuple
import argparse
import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

# def flash_mla_fwd_sp(
#     q: torch.Tensor,
#     k_cache: torch.Tensor,
#     block_table: torch.Tensor,
#     cache_seqlens: torch.Tensor,
#     head_dim_v: int,
#     tile_scheduler_metadata: torch.Tensor,
#     num_splits: torch.Tensor,
#     softmax_scale: Optional[float] = None,
#     causal: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Arguments:
#         q: (batch_size, num_heads_q, head_dim).
#         k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
#         block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
#         cache_seqlens: (batch_size), torch.int32.
#         head_dim_v: Head_dim of v.
#         tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
#         num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
#         softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
#         causal: bool. Whether to apply causal attention mask.

#     Return:
#         out: (batch_size, num_heads_q, head_dim_v).
#     """
#     # q: Tensor, dtype=torch.bfloat16, shape=(128, 1, 32, 576)
#     # k_cache: Tensor, dtype=torch.bfloat16, shape=(27662, 64, 1, 576)
#     # block_table: Tensor, dtype=torch.int32, shape=(128, 27662)
#     # cache_seqlens: Tensor, dtype=torch.int32, shape=(128,)
#     # head_dim_v: 512
#     # tile_scheduler_metadata:
#     # num_splits: num_splits: Tensor, dtype=torch.int32, shape=(129,)
#     # softmax_scale: 0.07216878364870322
#     # causal: True

#     # TODO: add all gather

#     out, lse = flash_mla_with_kvcache(
#         q,
#         k_cache,
#         block_table,
#         cache_seqlens,
#         head_dim_v,
#         tile_scheduler_metadata,
#         num_splits,
#         softmax_scale,
#         causal,
#     )

#     # TODO: add all2all

#     return out.squeeze(1)

if __name__ == "__main__":
    # -------- 1. 每个 RANK 独立指定 kv 长度 --------
    # 这里用 4 个 RANK，每个 RANK 4 条序列做演示
    kv_lens_per_rank = [
        [2048, 2048, 2048, 2048],   # rank0
        [2048, 2048, 2048, 2048],   # rank1
        [2048, 2048, 2048, 2048],   # rank2
        [2048, 2048, 2048, 2048],   # rank3
    ]

    parser = argparse.ArgumentParser(
        description="Benchmark / test script for attention-like workload."
    )
    args = parser.parse_args()
    args.num_ranks = 4          # 与上面列表长度保持一致
    args.rank = 0               # 实际运行时用 torchrun / mpirun 注入

    torch.cuda.set_device(args.rank)
    device = torch.device("cuda", args.rank)

    dtype = torch.bfloat16
    num_query_heads = 32
    head_size = 576
    head_dim_v = 512
    num_seqs = len(kv_lens_per_rank[args.rank])   # 按当前 RANK 的序列数
    softmax_scale = head_size ** -0.5

    # ---------------- 2. 构造张量 ----------------
    query = torch.randn(
        num_seqs, 1, num_query_heads, head_size, dtype=dtype, device=device
    ) / 10

    block_size = 64
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks
    num_kv_heads = 1

    key_value_cache = torch.randn(
        NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    key_value_cache_this_rank = key_value_cache[
        args.rank * NUM_BLOCKS_PER_RANK : (args.rank + 1) * NUM_BLOCKS_PER_RANK
    ]

    max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
    block_tables_list = [
        torch.randint(
            0,
            NUM_BLOCKS_PER_RANK,
            (num_seqs, max_num_blocks_per_seq_per_rank),
            dtype=torch.int32,
            device=device,
        )
        for _ in range(args.num_ranks)
    ]
    block_tables_this_rank = block_tables_list[args.rank]

    # --------- 3. 取当前 RANK 对应的 kv 长度 --------
    kv_lens_tensor = torch.tensor(
        kv_lens_per_rank[args.rank], dtype=torch.int32, device=device
    )

    # 下面的 global_kv_lens_tensor 只是为了后向兼容打印，可删
    global_kv_lens_tensor = torch.cat(
        [torch.tensor(kv_lens_per_rank[r], dtype=torch.int32, device=device).view(1, -1)
         for r in range(args.num_ranks)],
        dim=0,
    )
    kv_lens_tensor_this_rank = global_kv_lens_tensor[args.rank]

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        kv_lens_tensor_this_rank,
        num_query_heads // num_kv_heads,
        num_kv_heads,
    )

    print(f"[RANK {args.rank}] query.shape: {query.shape}", flush=True)
    print(f"[RANK {args.rank}] key_value_cache.shape: {key_value_cache.shape}", flush=True)
    print(f"[RANK {args.rank}] block_tables_this_rank.shape: {block_tables_this_rank.shape}", flush=True)

    out, lse = flash_mla_with_kvcache(
        query,
        key_value_cache,          # 仍传递全局 cache 即可
        block_tables_this_rank,
        kv_lens_tensor_this_rank,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        True,
    )

    print(f"[RANK {args.rank}] out.shape: {out.shape}", flush=True)
    print(f"[RANK {args.rank}] lse.shape: {lse.shape}", flush=True)

    


    