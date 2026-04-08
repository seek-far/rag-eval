from __future__ import annotations

import gc
import logging
from math import floor, log2
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceEncoderRuntime:
    """Shared SentenceTransformer runtime with optional multi-GPU encoding."""

    def __init__(
        self,
        model_name: str,
        primary_device: str,
        devices: list[str],
        batch_size: int,
        auto_batch: bool,
        batch_min: int,
        batch_max: int,
        batch_utilization: float,
        stage_name: str,
        keep_primary_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.primary_device = primary_device
        self.devices = devices or [primary_device]
        self.configured_batch_size = batch_size
        self.auto_batch = auto_batch
        self.batch_min = max(1, batch_min)
        self.batch_max = max(self.batch_min, batch_max)
        self.batch_utilization = min(max(batch_utilization, 0.1), 0.98)
        self.stage_name = stage_name

        load_device = "cpu" if self.uses_multi_device else self.primary_device
        self.model = SentenceTransformer(self.model_name, device=load_device)
        self._primary_model: SentenceTransformer | None = None
        self._pool: dict[str, Any] | None = None
        self._resolved_batches: dict[bool, int] = {}

        if self.uses_multi_device:
            self._pool = self.model.start_multi_process_pool(
                target_devices=self.devices
            )
            if keep_primary_model:
                self._primary_model = SentenceTransformer(
                    self.model_name,
                    device=self.primary_device,
                )

    @property
    def uses_multi_device(self) -> bool:
        return len(self.devices) > 1

    @property
    def device_label(self) -> str:
        return ",".join(self.devices)

    def close(self) -> None:
        if self._pool is not None:
            try:
                self.model.stop_multi_process_pool(self._pool)
            finally:
                self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        batch_size = self.resolve_batch_size(
            texts, normalize_embeddings=normalize_embeddings
        )
        return self._encode_with_batch(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

    def _encode_with_batch(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            "convert_to_numpy": True,
            "normalize_embeddings": normalize_embeddings,
        }
        if self._pool is not None:
            kwargs["pool"] = self._pool
        else:
            kwargs["device"] = self.primary_device
        return self.model.encode(texts, **kwargs)

    def encode_one(
        self,
        text: str,
        *,
        normalize_embeddings: bool,
    ):
        model = self._primary_model or self.model
        return model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            device=self.primary_device,
        )[0]

    def resolve_batch_size(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> int:
        if not self.auto_batch:
            return self.configured_batch_size

        cache_key = bool(normalize_embeddings)
        if cache_key not in self._resolved_batches:
            self._resolved_batches[cache_key] = self._probe_batch_size(
                texts,
                normalize_embeddings=normalize_embeddings,
            )
        return self._resolved_batches[cache_key]

    def _probe_batch_size(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> int:
        if not self._all_cuda_devices():
            logger.info(
                "%s auto batch skipped: non-CUDA device(s) %s, using %d",
                self.stage_name,
                self.device_label,
                self.configured_batch_size,
            )
            return self.configured_batch_size

        seed = [t for t in texts if isinstance(t, str) and t.strip()]
        if not seed:
            logger.info(
                "%s auto batch skipped: no sample texts, using %d",
                self.stage_name,
                self.configured_batch_size,
            )
            return self.configured_batch_size

        candidate = self._initial_batch_guess()
        logger.info(
            "%s auto batch probing on %s (start=%d, min=%d, max=%d)",
            self.stage_name,
            self.device_label,
            candidate,
            self.batch_min,
            self.batch_max,
        )

        if not self._try_batch(candidate, seed, normalize_embeddings=normalize_embeddings):
            failed = candidate
            while candidate > self.batch_min:
                candidate = max(self.batch_min, candidate // 2)
                if self._try_batch(candidate, seed, normalize_embeddings=normalize_embeddings):
                    break
                failed = candidate
            else:
                logger.warning(
                    "%s auto batch fell back to %d on %s",
                    self.stage_name,
                    self.batch_min,
                    self.device_label,
                )
                return self.batch_min
        else:
            failed = None
            while candidate < self.batch_max:
                next_candidate = min(self.batch_max, candidate * 2)
                if next_candidate == candidate:
                    break
                if self._try_batch(
                    next_candidate,
                    seed,
                    normalize_embeddings=normalize_embeddings,
                ):
                    candidate = next_candidate
                else:
                    failed = next_candidate
                    break

        if failed is not None and failed > candidate + 1:
            low, high = candidate, failed
            while low + 1 < high:
                mid = (low + high) // 2
                if self._try_batch(mid, seed, normalize_embeddings=normalize_embeddings):
                    low = mid
                else:
                    high = mid
            candidate = low

        logger.info(
            "%s auto batch selected %d on %s",
            self.stage_name,
            candidate,
            self.device_label,
        )
        return candidate

    def _try_batch(
        self,
        batch_size: int,
        seed_texts: list[str],
        *,
        normalize_embeddings: bool,
    ) -> bool:
        probe_texts = _repeat_to_length(seed_texts, batch_size)
        try:
            self._encode_with_batch(
                probe_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
            return True
        except RuntimeError as exc:
            if _is_oom_error(exc):
                logger.debug(
                    "%s auto batch rejected %d on %s: %s",
                    self.stage_name,
                    batch_size,
                    self.device_label,
                    exc,
                )
                self._clear_cuda_cache()
                return False
            raise
        finally:
            self._clear_cuda_cache()

    def _initial_batch_guess(self) -> int:
        free_gib = self._min_free_cuda_memory_gib()
        if free_gib is None:
            return self.configured_batch_size

        scaled = max(
            self.batch_min,
            min(self.batch_max, int(max(1.0, free_gib) * 4)),
        )
        if scaled <= 1:
            return 1
        return min(self.batch_max, max(self.batch_min, 2 ** floor(log2(scaled))))

    def _min_free_cuda_memory_gib(self) -> float | None:
        free_values: list[float] = []
        for device_name in self.devices:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_name)
            except Exception:
                return None
            free_values.append((free_bytes / 1024**3) * self.batch_utilization)
        return min(free_values) if free_values else None

    def _all_cuda_devices(self) -> bool:
        return bool(self.devices) and all(d.startswith("cuda") for d in self.devices)

    def _clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            for device_name in self.devices:
                if device_name.startswith("cuda"):
                    try:
                        with torch.cuda.device(device_name):
                            torch.cuda.empty_cache()
                    except Exception:
                        continue
        gc.collect()


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _repeat_to_length(items: list[str], target_len: int) -> list[str]:
    if not items:
        return []
    if len(items) >= target_len:
        return items[:target_len]
    repeated: list[str] = []
    while len(repeated) < target_len:
        remaining = target_len - len(repeated)
        repeated.extend(items[:remaining])
    return repeated
