from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_timestamp: str
    run_dir: Path
    verbose_log_path: Path
    eval_log_path: Path


class ProgressOnlyFilter(logging.Filter):
    """Allow only high-level progress messages on the console."""

    def filter(self, record):
        return record.name == "progress"


def _quiet_non_progress_console_handlers() -> None:
    """Prevent non-progress loggers from writing directly to terminal."""

    logger_dict = logging.root.manager.loggerDict
    for logger in logger_dict.values():
        if not isinstance(logger, logging.Logger):
            continue
        if logger.name == "progress":
            continue

        logger.handlers = [
            handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
        ]
        logger.propagate = True


def _install_non_progress_stream_handler_guard() -> None:
    """Prevent future non-progress stream handlers from attaching to terminal."""

    if getattr(logging, "_progress_guard_installed", False):
        return

    original_add_handler = logging.Logger.addHandler

    def guarded_add_handler(self, handler):
        is_stream = isinstance(handler, logging.StreamHandler)
        is_file = isinstance(handler, logging.FileHandler)
        if self.name != "progress" and is_stream and not is_file:
            return
        return original_add_handler(self, handler)

    logging.Logger.addHandler = guarded_add_handler
    logging._progress_guard_installed = True


def configure_run_logging(log_dir: str | Path = "logs") -> RunPaths:
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = log_dir / f"benchmark_{run_timestamp}"
    run_dir.mkdir(exist_ok=False)
    verbose_log_path = run_dir / "verbose.log"
    eval_log_path = run_dir / "evaluation.json"

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.handlers = [
        handler for handler in root_logger.handlers if not isinstance(handler, logging.StreamHandler)
    ]
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(ProgressOnlyFilter())
    root_logger.addHandler(console_handler)

    has_run_file_handler = any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", "") == str(verbose_log_path)
        for handler in root_logger.handlers
    )
    if not has_run_file_handler:
        file_handler = RotatingFileHandler(str(verbose_log_path), maxBytes=10485760, backupCount=5)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    progress_logger = logging.getLogger("progress")
    progress_logger.info("Benchmark run started")
    progress_logger.info("Run directory: %s", run_dir)
    progress_logger.info("Verbose log: %s", verbose_log_path)
    progress_logger.info("Evaluation log: %s", eval_log_path)

    _install_non_progress_stream_handler_guard()
    _quiet_non_progress_console_handlers()

    return RunPaths(
        run_timestamp=run_timestamp,
        run_dir=run_dir,
        verbose_log_path=verbose_log_path,
        eval_log_path=eval_log_path,
    )
