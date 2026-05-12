from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from npupy_xdna.dispatch.dtype_convert import convert_for_template
from npupy_xdna.dispatch.extract import numpy_op_to_region
from npupy_xdna.regions.region import Region
from npupy_xdna.heuristic.classifier import RegionClassifier
from npupy_xdna.heuristic.cost_model import CostModel
from npupy_xdna.heuristic.offload import OffloadHeuristic
from npupy_xdna.runtime.cpu_runner import CpuRunner
from npupy_xdna.runtime.npu_runner import NpuRunner
from npupy_xdna.templates.cgra import CgraTemplate
from npupy_xdna.templates.col_independent import ColIndependentTemplate
from npupy_xdna.templates.compute_pool import ComputePoolTemplate
from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate

logger = logging.getLogger(__name__)

_LOG_PATH = Path("npupy_xdna/results/dispatch.log")


class Dispatcher:
    def __init__(self) -> None:
        self.classifier = RegionClassifier()
        self.cost_model = CostModel()
        self.offload = OffloadHeuristic(self.cost_model, self.classifier)
        self.npu_runner = NpuRunner()
        self.cpu_runner = CpuRunner()
        self.templates: dict[str, Any] = {
            "gemm_fusion": GemmFusionTemplate(),
            "col_independent": ColIndependentTemplate(),
            "compute_pool": ComputePoolTemplate(),
            "cgra": CgraTemplate(),
        }
        self._log_path = _LOG_PATH

    def dispatch(
        self,
        orig_fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        info: Optional[dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        func_name_pre = getattr(orig_fn, "__name__", None)
        if func_name_pre == "tanh":
            first_arr = next((a for a in args if isinstance(a, np.ndarray)), None)
            if first_arr is None or first_arr.dtype != np.int16:
                self._log_entry(info, None, "tanh_non_int16_fallback")
                return None

        converted_args: list[Any] = []
        for idx, inp in enumerate(args):
            if not isinstance(inp, np.ndarray):
                converted_args.append(inp)
                continue
            try:
                converted, loss = convert_for_template(inp)
            except TypeError:
                converted_args.append(inp)
                continue
            if loss.would_overflow:
                self._log_entry(
                    info,
                    None,
                    f"dtype_overflow: input[{idx}] max_abs={loss.max_abs_input:.4g} > 32767, cpu_fallback",
                )
                return None
            if inp.dtype != np.int16:
                self._log_entry(
                    info,
                    None,
                    f"dtype_convert: input[{idx}] {inp.dtype}->int16 max_abs={loss.max_abs_input:.4g}",
                )
            converted_args.append(converted)

        try:
            region = numpy_op_to_region(orig_fn, tuple(converted_args), kwargs)
        except Exception as exc:  # pragma: no cover
            self._log_entry(info, None, f"extract_error: {exc}")
            return None

        if region is None:
            self._log_entry(info, None, "unsupported_op")
            return None

        try:
            decision = self.offload.decide(region)
        except Exception as exc:  # pragma: no cover
            self._log_entry(info, None, f"heuristic_error: {exc}")
            return None

        self._log_entry(info, decision, "heuristic_done")

        if decision.action == "cpu_fallback":
            return None

        template_name = decision.template
        template = self.templates.get(template_name)
        if template is None:
            self._log_entry(info, decision, f"unknown_template: {template_name}")
            return None

        try:
            configs = template.config_space(region)
        except Exception as exc:
            self._log_entry(info, decision, f"config_space_error: {exc}")
            return None

        if not configs:
            self._log_entry(info, decision, "empty_config_space")
            return None

        config = configs[0]

        from npupy_xdna.runtime.iron_jit import XCLBIN_CACHE_DIR, _cache_key
        shape_tuple = tuple(region.inputs[0].shape) if region.inputs else ()
        cache_key = _cache_key(template_name, shape_tuple)
        cache_path = XCLBIN_CACHE_DIR / f"{template_name}_{cache_key}.xclbin"
        cache_status = "xclbin_cache_hit" if cache_path.exists() else "xclbin_cache_miss"
        self._log_entry(info, decision, cache_status)

        try:
            iron_fn = template.lower(region, config)

            final_inputs = list(converted_args)
            if template_name == "gemm_fusion" and len(final_inputs) >= 2:
                final_inputs[1] = np.ascontiguousarray(final_inputs[1].T)

            result = self.npu_runner.run(
                region, config, iron_fn, final_inputs, timeout_s=60.0
            )
        except Exception as exc:
            self._log_entry(info, decision, f"npu_error: {exc}")
            return None

        if result.status == "ok":
            self._log_entry(info, decision, f"npu_ok latency={result.latency_us:.1f}us")
            return result.output

        self._log_entry(info, decision, f"npu_failed: {result.status}")
        return None

    def _log_entry(
        self,
        info: Optional[dict[str, Any]],
        decision: Any,
        outcome: str,
    ) -> None:
        try:
            log_path = Path(self._log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
            func = (info or {}).get("func", "?")
            shapes = [s.get("shape", ()) for s in (info or {}).get("arg_specs", [])]

            if decision is not None:
                action = getattr(decision, "action", "?")
                template = getattr(decision, "template", None)
                speedup = getattr(decision, "predicted_speedup", None)
                dec_str = (
                    f"action={action}"
                    + (f" template={template}" if template else "")
                    + (f" speedup={speedup:.2f}x" if speedup is not None else "")
                )
            else:
                dec_str = "decision=none"

            line = f"{ts} | func={func} shapes={shapes} | {dec_str} | {outcome}\n"
            with open(log_path, "a") as fh:
                fh.write(line)
        except Exception:  # pragma: no cover
            pass

    def dispatch_region(
        self,
        region: Region,
        inputs: list[np.ndarray],
        *,
        info: Optional[dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        try:
            decision = self.offload.decide(region)
        except Exception as exc:
            self._log_entry(info, None, f"heuristic_error: {exc}")
            return None

        self._log_entry(info, decision, "heuristic_done")

        if decision.action == "cpu_fallback":
            return None

        template_name = decision.template
        template = self.templates.get(template_name)
        if template is None:
            self._log_entry(info, decision, f"unknown_template: {template_name}")
            return None

        try:
            configs = template.config_space(region)
        except Exception as exc:
            self._log_entry(info, decision, f"config_space_error: {exc}")
            return None

        if not configs:
            self._log_entry(info, decision, "empty_config_space")
            return None

        config = configs[0]

        from npupy_xdna.runtime.iron_jit import XCLBIN_CACHE_DIR, _cache_key
        shape_tuple = tuple(region.inputs[0].shape) if region.inputs else ()
        cache_key = _cache_key(template_name, shape_tuple)
        cache_path = XCLBIN_CACHE_DIR / f"{template_name}_{cache_key}.xclbin"
        cache_status = "xclbin_cache_hit" if cache_path.exists() else "xclbin_cache_miss"
        self._log_entry(info, decision, cache_status)

        try:
            iron_fn = template.lower(region, config)
            result = self.npu_runner.run(region, config, iron_fn, inputs, timeout_s=60.0)
        except Exception as exc:
            self._log_entry(info, decision, f"npu_error: {exc}")
            return None

        if result.status == "ok":
            self._log_entry(info, decision, f"npu_ok latency={result.latency_us:.1f}us")
            return result.output

        self._log_entry(info, decision, f"npu_failed: {result.status}")
        return None
