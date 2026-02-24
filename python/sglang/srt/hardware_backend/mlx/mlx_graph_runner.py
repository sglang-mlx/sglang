# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Run the model with MLX mx.compile."""                                                                              
                                                                                                                    
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import mlx.core as mx

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

class MLXGraphRunner:
    """Runs forward pass using mx.compile for JIT optimization on Apple Silicon."""

    def __init__(self, model_runner: ModelRunner):
        # 1. Store model_runner ref, device
        self.model_runner = model_runner
        self.device = model_runner.device

        self.capture_hidden_mode = CaptureHiddenMode.NULL

        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        assert not model_runner.server_args.enable_lora, "enable_lora should be false"
        assert model_runner.spec_algorithm == SpeculativeAlgorithm.NONE, "spec_algorithm should be None"
        assert not model_runner.model_config.is_encoder_decoder, "is_encoder_decoder should be false"
        assert model_runner.server_args.dp_size == 1, "..."
        assert model_runner.server_args.pp_size == 1, "..."
        assert model_runner.server_args.tp_size == 1, "..."

        self.bs = None
        self._compiled_forward = mx.compile(model_runner.model.forward)
        self._warmup()

    def _warmup(self):
        """Pre-compile for common batch sizes to populate mx.compile's shape cache."""
        max_bs = min(8, self.model_runner.req_to_token_pool.size)
        warmup_batch_size = list(range(1, max_bs + 1))

        log_info_on_rank0(logger, f"Warming up mx.compile for batch sizes {warmup_batch_size}")

        import torch

        for bs in warmup_batch_size:
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=bs,
                input_ids=torch.zeros((bs,), dtype=torch.int64),
                positions=torch.zeros((bs,), dtype=torch.int64),
                req_pool_indices=torch.zeros((bs,), dtype=torch.int64),
                seq_lens=torch.ones((bs,), dtype=torch.int64),
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=torch.zeros((bs,), dtype=torch.int64),
                seq_lens_sum=bs,
                return_logprob=False,
                capture_hidden_mode=self.capture_hidden_mode,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                global_forward_mode=ForwardMode.DECODE,
            )
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
            self._compiled_forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )

            mx.eval()
        log_info_on_rank0(logger, "MLX warmup complete")

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """mx.compile handles any batch size on-demand, so always True for decode."""
        requested = max(
            forward_batch.capture_hidden_mode,
            getattr(forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL)
            or CaptureHiddenMode.NULL,
        )
        return (
            requested == CaptureHiddenMode.NULL
            or requested == self.capture_hidden_mode
        )

    def _recompile_if_needed(self, forward_batch: ForwardBatch):
        """If capture_hidden_mode changed, recreate mx.compile wrapper and re-warmup."""
        required = max(
            forward_batch.capture_hidden_mode,
            getattr(forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL),
            CaptureHiddenMode.FULL if self.model_runner.server_args.enable_return_hidden_states
            else CaptureHiddenMode.NULL,
        )
        if self.capture_hidden_mode != required:
            self.capture_hidden_mode = required
            self._compiled_forward = mx.compile(self.model_runner.model.forward)
            self._warmup()

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        """Execute the compiled forward pass."""

        assert pp_proxy_tensors is None, "PP not supported in MLXGraphRunner"

        self._recompile_if_needed(forward_batch)

        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        self.bs = forward_batch.batch_size

        return self._compiled_forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

