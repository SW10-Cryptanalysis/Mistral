import os
import sys
import json
import builtins
import pytest
from typing import Any
from unittest.mock import mock_open

# 1. Safely patch file I/O globally before any imports happen
_original_open = builtins.open
_original_exists = os.path.exists


def safe_open(file: Any, *args: Any, **kwargs: Any) -> Any:
    """Intercept ONLY metadata.json reads to prevent standard library crashes (like gettext)."""
    file_path = str(getattr(file, "name", file))
    if "metadata.json" in file_path:
        return mock_open(read_data=json.dumps({"max_symbol_id": 2503}))(
            file,
            *args,
            **kwargs,
        )
    return _original_open(file, *args, **kwargs)


def safe_exists(path: Any) -> bool:
    """Pretend metadata.json always exists for testing."""
    if "metadata.json" in str(path):
        return True
    return _original_exists(path)


# Inject the patches directly into builtins and os
builtins.open = safe_open
os.path.exists = safe_exists


# 2. Setup standard Pytest Arguments
def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI flags so pytest doesn't reject them."""
    parser.addoption(
        "--with-spaces",
        action="store_true",
        default=True,
        help="If enabled the model trains with space tokens in the training dataset",
    )
    parser.addoption(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset (e.g., 'tokenized_spaced', 'tokenized_normal_truncated_4000').",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Inject custom flags into sys.argv so argparse in config.py can find them."""
    # Use the exact flag name strings to avoid ValueError
    with_spaces = config.getoption("--with-spaces", default=None)
    dataset_path = config.getoption("--dataset-path", default=None)

    if with_spaces and "--with-spaces" not in sys.argv:
        sys.argv.append("--with-spaces")

    if dataset_path and "--dataset-path" not in sys.argv:
        # Strip literal single quotes if they leaked in from addopts
        clean_path = str(dataset_path).strip("'").strip('"')
        sys.argv.extend(["--dataset-path", clean_path])
