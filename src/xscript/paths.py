"""Central path layout.

Everything heavy lives on Lustre scratch (override with XSCRIPT_SCRATCH).
Small, committable outputs (tables, metrics) go to <repo>/results.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Isambard scratch layout is /scratch/<project>/<user> with user "<name>.<project>".
_user = os.environ.get("USER", "user")
_project = _user.rsplit(".", 1)[-1]
SCRATCH = Path(os.environ.get("XSCRIPT_SCRATCH", f"/scratch/{_project}/{_user}/xscript"))

RESULTS = Path(os.environ.get("XSCRIPT_RESULTS", REPO / "results"))

# scratch subtrees
FLORES_DIR = SCRATCH / "flores_plus"          # downloaded FLORES+ jsonl files
MANIFEST_CACHE = SCRATCH / "manifests"        # cached HF file/size listings (transient)
TOK_CORPORA = SCRATCH / "tok_corpora"         # tokenizer training corpora
TOKENIZERS = SCRATCH / "tokenizers"           # trained tokenizer artifacts
POOLS = SCRATCH / "pools"                     # per-language text pools (zstd jsonl)
SHARDS = SCRATCH / "shards"                   # per-(lang, tokenizer) uint16 token shards
RUNS = SCRATCH / "runs"                       # checkpoints + training logs
HOLDOUT = SCRATCH / "holdout"                 # in-domain eval holdout (excluded from pools)


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def tokenizer_dir(name: str) -> Path:
    return TOKENIZERS / name


def pool_dir(lang: str) -> Path:
    return POOLS / lang


def shard_dir(lang: str, tok: str) -> Path:
    return SHARDS / f"{lang}__{tok}"


def run_dir(run_name: str) -> Path:
    return RUNS / run_name
