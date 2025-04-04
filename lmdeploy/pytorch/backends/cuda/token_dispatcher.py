# Copyright (c) OpenMMLab. All rights reserved.
try:
    from deep_ep import Buffer

    use_deepep = True
except ImportError:
    use_deepep = False

from typing import List, Tuple

import torch
import torch.distributed as dist

from ..default.token_dispatcher import AlltoAllTokenDispatcher
from ..token_dispatcher import TokenDispatcherImpl

_buffer_normal = None


def get_buffer_normal(group: dist.ProcessGroup, hidden_bytes: int):
    """Copy from DeepEP example usage in model inference prefilling.

    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-model-training-or-inference-prefilling
    """
    global _buffer_normal
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
    if (_buffer_normal is None or _buffer_normal.group != group or _buffer_normal.num_nvl_bytes < num_nvl_bytes
            or _buffer_normal.num_rdma_bytes < num_rdma_bytes):
        _buffer_normal = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer_normal


class DeepEPTokenDispatcher(TokenDispatcherImpl):
    """Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-
    LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_bytes = params_dtype.itemsize
        # Handle used for combine operation
        self.handle = None
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.buffer_normal = get_buffer_normal(self.group, self.hidden_size * self.params_bytes)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_list: List[int] = None,
        previous_event=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        (
            hidden_states,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.dispatch_normal(hidden_states, topk_idx, topk_weights, self.num_experts, previous_event)
        self.tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            device=hidden_states.device,
            dtype=torch.int64,
        )
        tokens_per_expert = self.get_number_of_tokens_per_expert()
        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        if hidden_states.shape[0] > 0:
            hidden_states = self.get_permuted_hidden_states_by_experts(hidden_states)
        return hidden_states, topk_idx, topk_weights, tokens_per_expert

    def dispatch_normal(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        previous_event=None,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer_normal.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.buffer_normal.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights.to(torch.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            hidden_states = self.get_restored_hidden_states_by_experts(hidden_states)
        hidden_states, event = self.combine_normal(hidden_states, self.handle)
        self.handle = None
        return hidden_states.view(self.hidden_shape)

    def combine_normal(self, x: torch.Tensor, handle: Tuple, previous_event=None):
        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.topk_idx, self.topk_weights

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """Get the number of tokens per expert."""
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.dispatched_routing_map, self.topk_weights = super().indices_to_multihot(
            self.topk_idx, self.topk_weights, self.num_experts)
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = super().permute(
            hidden_states,
            self.dispatched_routing_map,
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        assert (self.topk_weights.dtype == torch.float32), 'DeepEP only supports float32 probs'
        hidden_states = super().unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.topk_weights,
        )
        return hidden_states.to(input_dtype)


class TokenDispatcherBuilder:
    """token dispatcher builder."""

    @staticmethod
    def build(
        group,
        num_experts,
        num_local_experts,
        hidden_size,
        params_dtype,
    ) -> TokenDispatcherImpl:
        """build."""
        if use_deepep is True:
            return DeepEPTokenDispatcher(
                group,
                num_experts,
                num_local_experts,
                hidden_size,
                params_dtype,
            )
        else:
            return AlltoAllTokenDispatcher(
                group,
                num_experts,
                num_local_experts,
            )
