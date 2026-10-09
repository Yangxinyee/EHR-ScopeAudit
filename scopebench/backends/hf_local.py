"""transformers backend: the frozen ScopePair `CudaOnlyModel` behaviour (CUDA only, greedy, one at a time).
Used for parity checks and as the fallback path."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import Generation, Request


class HFLocalBackend:
    name = "hf_local"

    def __init__(self, model_path: str | Path, seed: int = 2027, enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA is required. Refusing CPU inference.")
        torch.cuda.set_device(0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(0)
        self.torch = torch
        self.model_path = str(model_path)
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto", device_map={"": 0}, low_cpu_mem_usage=True)
        self.model.eval()
        devices = {p.device.type for p in self.model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"CPU/offloaded parameters detected: {sorted(devices)}")
        self.device = next(self.model.parameters()).device

    def _generate(self, req: Request) -> Generation:
        kwargs = {}
        try:
            prompt = self.tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=False, use_cache=True, pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        self.torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - started
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return Generation(text=text, tokens_in=int(inputs["input_ids"].shape[1]), tokens_out=int(generated.shape[0]), latency_seconds=elapsed)

    def generate_many(self, requests: list[Request]) -> dict[str, Generation]:
        return {r.key: self._generate(r) for r in requests}

    def metadata(self) -> dict[str, Any]:
        props = self.torch.cuda.get_device_properties(0)
        return {
            "backend": self.name,
            "model_path": self.model_path,
            "seed": self.seed,
            "torch_version": self.torch.__version__,
            "cuda_runtime": self.torch.version.cuda,
            "gpu_name": props.name,
            "gpu_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
            "cpu_inference_allowed": False,
        }
