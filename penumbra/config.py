import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 8790

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memory"

    @property
    def ledger_dir(self) -> Path:
        """Legacy: the retired pattern-memory database and the old inject / recall ledger (kept as history, never read)."""
        return self.data_dir / "ledger"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.sqlite"


def load_config(data_dir: str | os.PathLike | None = None, port: int | None = None) -> Config:
    data = Path(data_dir or os.environ.get("PENUMBRA_DATA") or PACKAGE_ROOT / "data").resolve()
    return Config(
        data_dir=data,
        host=os.environ.get("PENUMBRA_HOST") or "127.0.0.1",
        port=int(port or os.environ.get("PENUMBRA_PORT") or 8790),
    )
