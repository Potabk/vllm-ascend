#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from typing import List, Optional

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context


class NPUCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: dist.ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[dist.ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        # TODO(hz): Refer to CudaCommunicator's implementation to integrate PyHcclCommunicator
        # init device according to rank
        self.device = torch.npu.current_device()

        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)

    def get_dp_rank(self) -> int:
        return get_dp_group().rank_in_group

    def get_dp_world_size(self) -> int:
        return get_dp_group().world_size

    def all_to_all(self,
                   input_: torch.Tensor,
                   scatter_dim: int = 0,
                   gather_dim: int = -1,
                   scatter_sizes: Optional[List[int]] = None,
                   gather_sizes: Optional[List[int]] = None) -> torch.Tensor:

        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [
                t.contiguous()
                for t in torch.split(input_, scatter_sizes, scatter_dim)
            ]
            output_list = []
            tensor_shape_base = input_list[self.rank].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(
                    torch.empty(tensor_shape,
                                dtype=input_.dtype,
                                device=input_.device))

        else:
            input_list = [
                t.contiguous() for t in torch.tensor_split(
                    input_, self.world_size, scatter_dim)
            ]
            output_list = [
                torch.empty_like(input_list[i]) for i in range(self.world_size)
            ]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        dp_rank = self.get_dp_rank()
        start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
        end = cu_tokens_across_dp_cpu[dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(self.get_dp_world_size()):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            get_dp_group().broadcast(buffer[start:end, :], idx)

        return buffer

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu

        hidden_states = self.naive_multicast(hidden_states,
                                             cu_tokens_across_dp_cpu)
        router_logits = self.naive_multicast(router_logits,
                                             cu_tokens_across_dp_cpu)
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu
        dp_rank = self.get_dp_rank()
        start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
        end = cu_tokens_across_dp_cpu[dp_rank]

        all_hidden_states = get_dp_group().all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states
