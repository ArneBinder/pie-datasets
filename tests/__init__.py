from pathlib import Path
from typing import Any, Dict

from datasets import DownloadMode, load_dataset

TESTS_ROOT = Path(__file__).parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures"
DATASET_BUILDERS_ROOT = Path("dataset_builders")


def _config_to_str(cfg: Dict[str, Any]) -> str:
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


def _check_hf_conll2003_is_available():
    try:
        load_dataset("conll2003", download_mode=DownloadMode.FORCE_REDOWNLOAD)
        return True
    except ConnectionError:
        return False
