import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def kernel_inter_rank_gqa_fwd_batch_decode_combine_kv(
    Mid_O,
    o,
    RankMask,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        mask_val = tl.load(RankMask + split_kv_id)

        if mask_val > 0:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)

            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )

# 测试函数：验证一维rank_mask的功能
def test_1d_rank_mask():
    # 1. 配置测试参数
    batch = 2               # 批大小
    num_q_heads = 2         # 查询头数量
    num_ranks = 3           # KV拆分的RANK数量
    v_head_dim = 4          # 值头维度(Lv)
    block_dv = 4            # 必须 >= v_head_dim
    
    # 2. 构造输入数据
    torch.manual_seed(42)  # 固定随机种子，保证结果可复现
    # Mid_O: shape=(num_ranks, batch, num_q_heads, v_head_dim + 1)
    mid_o = torch.randn(num_ranks, batch, num_q_heads, v_head_dim + 1, device='cuda', dtype=torch.float32)
    
    # 3. 构造一维rank_mask配置（关键修改）
    # 配置1: 全1掩码（所有RANK都参与计算）
    rank_mask_all_1 = torch.ones(num_ranks, device='cuda', dtype=torch.int32)
    # 配置2: 部分RANK掩码为0（RANK=1不参与计算）
    rank_mask_partial_0 = torch.ones(num_ranks, device='cuda', dtype=torch.int32)
    rank_mask_partial_0[1] = 0  # 一维掩码，对所有batch生效
    
    # 4. 分配输出张量
    output_all_1 = torch.empty((batch, num_q_heads, v_head_dim), device='cuda', dtype=torch.float32)
    output_partial_0 = torch.empty((batch, num_q_heads, v_head_dim), device='cuda', dtype=torch.float32)
    
    # 5. 启动kernel计算
    grid = (batch, num_q_heads, 1)
    
    # 配置1计算
    kernel_inter_rank_gqa_fwd_batch_decode_combine_kv[grid](
        mid_o, output_all_1, rank_mask_all_1,
        batch, num_q_heads,
        mid_o.stride(1),  # stride_mid_ob (batch维度步长)
        mid_o.stride(2),  # stride_mid_oh (head维度步长)
        mid_o.stride(0),  # stride_mid_os (num_ranks维度步长)
        output_all_1.stride(0),  # stride_obs
        output_all_1.stride(1),  # stride_oh
        num_ranks, block_dv, v_head_dim
    )
    
    # 配置2计算
    kernel_inter_rank_gqa_fwd_batch_decode_combine_kv[grid](
        mid_o, output_partial_0, rank_mask_partial_0,
        batch, num_q_heads,
        mid_o.stride(1), mid_o.stride(2), mid_o.stride(0),
        output_partial_0.stride(0), output_partial_0.stride(1),
        num_ranks, block_dv, v_head_dim
    )
    
    # 6. 手动计算预期结果（仅使用有效RANK：0和2）
    valid_ranks = [0, 2]
    tv_list = [mid_o[rank, ..., :v_head_dim] for rank in valid_ranks]
    logic_list = [mid_o[rank, ..., v_head_dim] for rank in valid_ranks]
    
    expected = []
    for b in range(batch):
        batch_result = []
        for h in range(num_q_heads):
            # 提取当前batch和head的logic值
            logics = [logic_list[r][b, h].item() for r in range(len(valid_ranks))]
            e_max = max(logics)
            
            # 计算exp(logic - e_max)
            exp_vals = [torch.exp(torch.tensor(l) - e_max) for l in logics]
            e_sum = sum(exp_vals)
            
            # 计算加权平均
            weighted_sum = torch.zeros(v_head_dim, device='cuda')
            for r in range(len(valid_ranks)):
                weighted_sum += exp_vals[r] * tv_list[r][b, h]
            batch_result.append(weighted_sum / e_sum)
        expected.append(torch.stack(batch_result))
    expected = torch.stack(expected)
    
    # 7. 打印结果并验证
    print("=== 一维rank_mask测试结果 ===")
    print(f"全1掩码输出（batch 0, head 0）:\n{output_all_1[0, 0]}")
    print(f"排除RANK=1的输出（batch 0, head 0）:\n{output_partial_0[0, 0]}")
    print(f"手动计算的预期输出（batch 0, head 0）:\n{expected[0, 0]}")
    
    # 验证不同batch是否受到相同掩码影响（应全部排除RANK=1）
    torch.testing.assert_close(
        output_partial_0, expected,
        rtol=1e-5, atol=1e-5,
        msg="排除RANK的计算结果与预期不符"
    )
    print("\n测试通过：一维rank_mask功能正常")

if __name__ == "__main__":
    test_1d_rank_mask()