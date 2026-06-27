# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import json
import logging
import numbers
import os
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .random_str import get_random_string

import pcre
from logbar import LogBar


_ANSI_ESCAPE_RE = pcre.compile(r"\x1b\[[0-9;]*m")


class _SilentProgress:
    """Minimal no-op progress handle for non-interactive test sessions."""

    def __init__(self, iterable=None):
        self._iterable = iterable if iterable is not None else ()
        self.current_iter_step = 0

    def __iter__(self):
        if isinstance(self._iterable, int):
            return iter(range(self._iterable))
        return iter(self._iterable)

    def __len__(self):
        if isinstance(self._iterable, int):
            return self._iterable
        return len(self._iterable)

    def attach(self, *_args, **_kwargs):
        return self

    def manual(self):
        return self

    def set(self, **_kwargs):
        return self

    def title(self, *_args, **_kwargs):
        return self

    def subtitle(self, *_args, **_kwargs):
        return self

    def draw(self, force: bool = False):
        return self

    def refresh(self):
        return self

    def next(self, step: int = 1):
        self.current_iter_step += int(step)
        return self

    def close(self):
        return None


class _AdaptiveLoggerProxy:
    """Proxy that keeps structured logs while adapting live rendering at call time."""

    def __init__(self, logger: LogBar):
        self._logger = logger

    def pb(self, iterable, *, output_interval: Optional[int] = None):
        if _suppress_live_renderables():
            return _SilentProgress(iterable)
        return self._logger.pb(iterable, output_interval=output_interval)

    def spinner(self, title: str = "", *, interval: float = 0.5, tail_length: int = 4):
        if _suppress_live_renderables():
            return _SilentProgress()
        return self._logger.spinner(title=title, interval=interval, tail_length=tail_length)

    def __getattr__(self, name):
        return getattr(self._logger, name)


def _suppress_live_renderables() -> bool:
    """Disable live progress redraws under non-interactive pytest capture."""

    if "PYTEST_CURRENT_TEST" not in os.environ:
        return False

    try:
        return not sys.stdout.isatty()
    except Exception:
        return True


def live_renderables_suppressed() -> bool:
    """Report whether redraw-based progress should be replaced by durable logs."""

    return _suppress_live_renderables()


_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "datasets",
    "filelock",
    "accelerate",
)

_THIRD_PARTY_LOGGING_CONFIGURED = False


def _configure_third_party_logging() -> None:
    """Suppress noisy HTTP/progress logs unless GPTQMODEL_VERBOSE is set."""
    global _THIRD_PARTY_LOGGING_CONFIGURED
    if _THIRD_PARTY_LOGGING_CONFIGURED:
        return
    _THIRD_PARTY_LOGGING_CONFIGURED = True

    verbose = os.environ.get("GPTQMODEL_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
    if verbose:
        return

    level = logging.WARNING
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:
        pass


def layer_dashboard_enabled() -> bool:
    """Return True when the per-layer in-place quant dashboard should be used."""
    flag = os.environ.get("GPTQMODEL_LAYER_DASHBOARD", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if live_renderables_suppressed():
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


LAYER_DASHBOARD_COLUMNS: tuple[str, ...] = (
    "method",
    "module",
    "loss",
    "nsamples",
    "damp",
    "time_s",
)


class LayerQuantDashboard:
    """In-place terminal summary refreshed after each layer (top/nvidia-smi style)."""

    _STAT_KEY_MAP = {
        "method": "method",
        "module": "module",
        "loss": "loss",
        "nsamples": "nsamples",
        "damp": "damp_percent",
        "time_s": "time",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines_printed = 0

    def close(self) -> None:
        """Leave the terminal on a fresh line after the last in-place redraw."""
        with self._lock:
            if self._lines_printed > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._lines_printed = 0

    def render(
        self,
        *,
        layer_index: int,
        layer_label: str,
        rows: Sequence[Dict[str, Any]],
    ) -> None:
        if not rows:
            return

        display_rows: list[list[str]] = []
        for stat in rows:
            display_rows.append([self._cell(stat, column) for column in LAYER_DASHBOARD_COLUMNS])

        header = f"Layer {layer_index} ({layer_label}) — {len(rows)} module(s)"
        table = render_table(
            display_rows,
            headers=list(LAYER_DASHBOARD_COLUMNS),
            tablefmt="grid",
        )
        output = f"{header}\n{table}\n"

        with self._lock:
            if self._lines_printed > 0:
                sys.stdout.write(f"\033[{self._lines_printed}A\033[J")
            sys.stdout.write(output)
            sys.stdout.flush()
            self._lines_printed = output.count("\n")

    @classmethod
    def _cell(cls, stat: Dict[str, Any], column: str) -> str:
        for key in (column, cls._STAT_KEY_MAP.get(column, "")):
            if not key:
                continue
            if key in stat and stat[key] not in (None, ""):
                return str(stat[key])
        legacy_keys = {
            "method": "method",
            "module": "module",
            "loss": "loss",
            "nsamples": "nsamples",
            "damp": "damp_percent",
            "time_s": "time",
        }
        from ..models.writer import (
            PROCESS_LOG_MODULE,
            PROCESS_LOG_NAME,
            PROCESS_LOG_TIME,
            QUANT_LOG_DAMP,
            QUANT_LOG_LOSS,
            QUANT_LOG_NSAMPLES,
        )

        writer_map = {
            "method": PROCESS_LOG_NAME,
            "module": PROCESS_LOG_MODULE,
            "loss": QUANT_LOG_LOSS,
            "nsamples": QUANT_LOG_NSAMPLES,
            "damp": QUANT_LOG_DAMP,
            "time_s": PROCESS_LOG_TIME,
        }
        writer_key = writer_map.get(column)
        if writer_key and writer_key in stat:
            return str(stat[writer_key])
        fallback = legacy_keys.get(column, column)
        value = stat.get(fallback, "")
        return "" if value is None else str(value)


REGION_TIMING_COLUMNS: tuple[str, ...] = (
    "region",
    "count",
    "last_s",
    "avg_s",
    "total_s",
    "pct",
    "source",
)

_PROJECT_LOG_DIR = Path("logs")


class RegionTimingDashboard:
    """In-place terminal summary for cumulative region timing (top/nvidia-smi style)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines_printed = 0

    def close(self) -> None:
        """Leave the terminal on a fresh line after the last in-place redraw."""
        with self._lock:
            if self._lines_printed > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._lines_printed = 0

    def render(
        self,
        *,
        title: str,
        rows: Sequence[Sequence[str]],
        headers: Optional[Sequence[str]] = None,
    ) -> None:
        if not rows:
            return

        column_headers = list(headers or REGION_TIMING_COLUMNS)
        table = render_table(rows, headers=column_headers, tablefmt="grid")
        output = f"{title}\n{table}\n"

        with self._lock:
            if self._lines_printed > 0:
                sys.stdout.write(f"\033[{self._lines_printed}A\033[J")
            sys.stdout.write(output)
            sys.stdout.flush()
            self._lines_printed = output.count("\n")


def setup_logger():
    _configure_third_party_logging()
    return _AdaptiveLoggerProxy(LogBar.shared())


def _table_cell_text(value: Any, floatfmt: Optional[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if floatfmt is not None and isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return format(float(value), floatfmt)
    return str(value)


def _visible_width(value: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", value))


def _pad_table_cell(value: str, width: int) -> str:
    return value + (" " * max(0, width - _visible_width(value)))


def _render_grid_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]) -> str:
    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row_line(values: Sequence[str]) -> str:
        return "| " + " | ".join(_pad_table_cell(value, widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [border(), row_line(headers), border()]
    lines.extend(row_line(row) for row in rows)
    lines.append(border())
    return "\n".join(lines)


def _render_github_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]) -> str:
    def row_line(values: Sequence[str]) -> str:
        return "| " + " | ".join(_pad_table_cell(value, widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [row_line(headers), separator]
    lines.extend(row_line(row) for row in rows)
    return "\n".join(lines)


def _render_simple_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]) -> str:
    def row_line(values: Sequence[str]) -> str:
        return "  ".join(_pad_table_cell(value, widths[idx]) for idx, value in enumerate(values))

    separator = "  ".join("-" * width for width in widths)
    lines = [row_line(headers), separator]
    lines.extend(row_line(row) for row in rows)
    return "\n".join(lines)


def render_table(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[Any],
    tablefmt: str = "grid",
    floatfmt: Optional[str] = None,
    logger: Optional[LogBar] = None,
) -> str:
    """Render a small diagnostic table using LogBar-compatible column sizing."""

    header_text = [str(header) for header in headers]
    row_text: list[list[str]] = []
    for row in rows:
        values = list(row)
        if len(values) != len(header_text):
            raise ValueError(
                f"Row length {len(values)} does not match header length {len(header_text)}"
            )
        row_text.append([_table_cell_text(value, floatfmt) for value in values])

    widths = [_visible_width(header) for header in header_text]
    if header_text:
        columns = (logger or LogBar.shared()).columns(
            cols=[{"label": header, "width": "fit"} for header in header_text],
            padding=1,
        )
        for row in row_text:
            columns.info.simulate(*row)
        widths = [max(widths[idx], width) for idx, width in enumerate(columns.widths)]

    for row in row_text:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], _visible_width(cell))

    tablefmt_normalized = (tablefmt or "grid").lower()
    if tablefmt_normalized == "github":
        return _render_github_table(header_text, row_text, widths)
    if tablefmt_normalized == "simple":
        return _render_simple_table(header_text, row_text, widths)
    return _render_grid_table(header_text, row_text, widths)


class QuantizationRegionTimer:
    """Aggregate and display timing statistics for key quantization stages."""

    DEFAULT_REGIONS = [
        ("model_load", "Model load"),
        ("model_reload", "Turtle reload"),
        ("capture_inputs", "Capture inputs"),
        ("forward_hook", "Forward hook"),
        ("pre_quant_forward", "Pre-quant forward"),
        ("process_quant", "Process quant"),
        ("post_quant_forward", "Post-quant replay"),
        ("submodule_finalize", "Submodule finalize"),
        ("submodule_finalize_create", "Finalize create"),
        ("submodule_finalize_pack", "Finalize pack"),
        ("submodule_finalize_offload", "Finalize offload"),
        ("process_finalize", "Process finalize"),
        ("model_save", "Model save"),
    ]

    def __init__(self, logger: Optional[LogBar] = None):
        self.logger = logger or setup_logger()
        self._lock = threading.Lock()
        self._region_labels: "OrderedDict[str, str]" = OrderedDict(self.DEFAULT_REGIONS)
        self._stats: "OrderedDict[str, Dict[str, float | int | str | None]]" = OrderedDict()
        self._pending_refresh = False
        self._dashboard = RegionTimingDashboard() if layer_dashboard_enabled() else None
        self._log_file: Optional[Path] = None
        self._flush_count = 0
        self.reset()

    @staticmethod
    def _build_log_file_path(current_time: str) -> Path:
        _PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        return _PROJECT_LOG_DIR / f"region_timing_log_{get_random_string()}_time_{current_time}.log"

    def reset(self) -> None:
        """Reset accumulated timing data."""
        with self._lock:
            if self._dashboard is not None:
                self._dashboard.close()
            self._stats = OrderedDict(
                (region, self._fresh_stat()) for region in self._region_labels.keys()
            )
            self._pending_refresh = False
            self._flush_count = 0
            current_time = datetime.now().strftime("%m_%d_%Y_%Hh_%Mm_%Ss")
            self._log_file = self._build_log_file_path(current_time)

    def _fresh_stat(self) -> Dict[str, float | int | None]:
        return {"total": 0.0, "count": 0, "last": 0.0, "source": None}

    def _populated_regions_locked(
        self,
    ) -> List[Tuple[str, Dict[str, float | int | str | None]]]:
        populated = [
            (region, stat)
            for region, stat in self._stats.items()
            if stat.get("count", 0)
        ]
        populated.sort(key=lambda item: float(item[1].get("total", 0.0)), reverse=True)
        return populated

    def _rows_locked(
        self,
        populated: Sequence[Tuple[str, Dict[str, float | int | str | None]]],
    ) -> Tuple[List[List[str]], float]:
        overall_total = sum(float(stat.get("total", 0.0)) for _, stat in populated)
        if overall_total <= 0:
            overall_total = 0.0

        rows: List[List[str]] = []
        for region, stat in populated:
            display_name = self._region_labels.get(region, region)
            total = float(stat.get("total", 0.0))
            count = int(stat.get("count", 0))
            last = float(stat.get("last", 0.0))
            avg = total / count if count else 0.0
            pct = (total / overall_total * 100.0) if overall_total > 0 else 0.0
            source = stat.get("source") or ""
            rows.append(
                [
                    display_name,
                    str(count),
                    f"{last:.3f}",
                    f"{avg:.3f}",
                    f"{total:.3f}",
                    f"{pct:.1f}%",
                    str(source),
                ]
            )
        return rows, overall_total

    def record(self, region: str, duration: float, *, source: Optional[str] = None) -> None:
        """Record a timing sample for a region and emit an updated summary."""

        if duration is None:
            return

        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            return

        if duration_value < 0:
            duration_value = 0.0

        with self._lock:
            if region not in self._stats:
                if region not in self._region_labels:
                    self._region_labels[region] = region.replace("_", " ").title()
                self._stats[region] = self._fresh_stat()

            stat = self._stats[region]
            stat["total"] = float(stat.get("total", 0.0)) + duration_value
            stat["count"] = int(stat.get("count", 0)) + 1
            stat["last"] = duration_value
            if source is not None:
                stat["source"] = source

            self._pending_refresh = True

    def flush(
        self,
        *,
        layer_index: Optional[int] = None,
        layer_label: Optional[str] = None,
        render: bool = True,
    ) -> None:
        """Refresh the in-place dashboard and append a JSON snapshot to the log file."""
        with self._lock:
            if not self._pending_refresh:
                return
            self._emit_locked(
                layer_index=layer_index,
                layer_label=layer_label,
                render=render,
            )
            self._pending_refresh = False

    def _emit_locked(
        self,
        *,
        layer_index: Optional[int] = None,
        layer_label: Optional[str] = None,
        render: bool = True,
    ) -> None:
        populated = self._populated_regions_locked()
        if not populated:
            return

        rows, _ = self._rows_locked(populated)
        headers = list(REGION_TIMING_COLUMNS)

        if render and self._dashboard is not None:
            title_parts = ["Region timing"]
            if layer_index is not None:
                label = layer_label or str(layer_index)
                title_parts.append(f"— Layer {layer_index} ({label})")
            self._dashboard.render(
                title=" ".join(title_parts),
                rows=rows,
                headers=headers,
            )

        self._append_snapshot_locked(
            layer_index=layer_index,
            layer_label=layer_label,
            populated=populated,
        )

    def _append_snapshot_locked(
        self,
        *,
        layer_index: Optional[int],
        layer_label: Optional[str],
        populated: Sequence[Tuple[str, Dict[str, float | int | str | None]]],
    ) -> None:
        if self._log_file is None:
            return

        snapshot = {
            "flush": self._flush_count,
            "layer_index": layer_index,
            "layer_label": layer_label,
            "regions": {
                region: {
                    "total": float(stat.get("total", 0.0)),
                    "count": int(stat.get("count", 0)),
                    "last": float(stat.get("last", 0.0)),
                    "source": stat.get("source"),
                }
                for region, stat in populated
            },
        }
        self._flush_count += 1

        with open(self._log_file, "a", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
            handle.write("\n")

    def write_full_log_summary(self) -> Optional[str]:
        """Write a human-readable summary table to disk and report its path once."""
        with self._lock:
            populated = self._populated_regions_locked()
            if not populated:
                if self._dashboard is not None:
                    self._dashboard.close()
                return None

            rows, overall_total = self._rows_locked(populated)
            log_file = self._log_file
            flush_count = self._flush_count

        if log_file is None:
            return None

        summary_path = log_file.with_suffix(".summary.log")
        table_text = render_table(rows, headers=list(REGION_TIMING_COLUMNS), tablefmt="grid")

        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("# Region timing summary\n")
            handle.write(f"# JSON log: {log_file}\n")
            handle.write(f"# Flushes: {flush_count}\n")
            handle.write(f"# Total tracked time: {overall_total:.3f}s\n\n")
            handle.write(table_text)
            handle.write("\n")

        if self._dashboard is not None:
            self._dashboard.render(title="Region timing (final)", rows=rows)
            self._dashboard.close()

        self.logger.info(
            f"Region timing log written to: {summary_path} (JSON snapshots: {log_file})"
        )
        return str(summary_path)

    def close(self) -> None:
        """Release any live terminal dashboard state."""
        if self._dashboard is not None:
            self._dashboard.close()

    @contextlib.contextmanager
    def measure(self, region: str, *, source: Optional[str] = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record(region, duration, source=source)

    def snapshot(self) -> Dict[str, Dict[str, float | int | str | None]]:
        with self._lock:
            return {
                region: {
                    "total": float(stat.get("total", 0.0)),
                    "count": int(stat.get("count", 0)),
                    "last": float(stat.get("last", 0.0)),
                    "source": stat.get("source"),
                }
                for region, stat in self._stats.items()
            }


@contextlib.contextmanager
def log_time_block(
    block_name: str,
    *,
    logger: Optional[LogBar] = None,
    module_name: Optional[str] = None,
) -> Iterator[None]:
    """Log the elapsed time of a block to the shared logger."""

    if logger is None:
        logger = setup_logger()

    start = time.perf_counter()
    try:
        yield
    finally:
        time.perf_counter() - start
        #logger.info(f"[time] {label} took {duration:.3f}s")
