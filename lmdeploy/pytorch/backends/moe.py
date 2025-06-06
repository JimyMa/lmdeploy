# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributed as dist


class SoftmaxTopKImpl(ABC):
    """Softmax topk implementation api."""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """forward."""
        raise NotImplementedError


class SoftmaxTopKBuilder(ABC):
    """Softmax topk implementation builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        raise NotImplementedError


class FusedMoEImpl(ABC):
    """Fused moe implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update weights."""
        return gate_up_weights, down_weights

    def support_ep(self):
        """Support expert parallelism."""
        return False

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        raise NotImplementedError


class FusedMoEBuilder(ABC):
    """Fused moe builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """Build from mlp."""
        raise NotImplementedError


class FusedMoEW8A8Impl(ABC):
    """Fused moe w8a8 implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """Update weights."""
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def support_ep(self):
        """Support expert parallelism."""
        return False

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                input_scale: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        raise NotImplementedError


class FusedMoEW8A8Builder(ABC):
    """Fused moe w8a8 builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              out_dtype: torch.dtype = torch.float16,
              quant_dtype: torch.dtype = torch.int8):
        """Build from mlp."""
        raise NotImplementedError


class FusedMoEBlockedF8Impl(ABC):
    """Fused moe blocked f8 implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """Update weights."""
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def support_ep(self):
        """Support expert parallelism."""
        return False

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                input_scale: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        raise NotImplementedError


class FusedMoEBlockedF8Builder(ABC):
    """Fused moe blocked f8 builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int,
              num_experts: int,
              hidden_dim: int = 1,
              renormalize: bool = False,
              block_size: int = 128,
              ep_size: int = 1,
              ep_group: dist.ProcessGroup = None,
              out_dtype: torch.dtype = torch.float16):
        """Build from mlp."""
        raise NotImplementedError
