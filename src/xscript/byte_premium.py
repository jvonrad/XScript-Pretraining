"""Byte-premium calibration on FLORES+ (thesis-plan step 3).

premium(L) = total UTF-8 bytes of L / total UTF-8 bytes of English, summed
over the *same* parallel sentences. Computed on the exact FLORES+ split(s) we
use everywhere else, so the calibration is self-consistent and reproducible
(plan caveat #1). Text is used exactly as distributed (no Unicode
normalization), matching how the corpora are consumed downstream.

We also fetch Arnett/Chang/Bergen's precomputed premiums
(github.com/catherinearnett/byte-premium-tool) as an external sanity check
(plan caveat #2); ours drive the pipeline.
"""
import json
import urllib.request
from pathlib import Path

from . import flores
from .langs import LANGS, ANCHOR
from .paths import RESULTS, ensure

ARNETT_REPO = "catherinearnett/byte-premium-tool"
# ISO 639-3 codes used by the Arnett tool for our languages.
ARNETT_CODES = {"en": "eng", "de": "deu", "fr": "fra", "ar": "arb", "zh": "cmn"}


def compute(splits=("dev", "devtest")) -> dict:
    langs = list(LANGS)
    per_split = {}
    tot = {l: 0 for l in langs}
    for split in splits:
        par = flores.load_parallel(langs, split)
        n = len(par[ANCHOR])
        b = {l: sum(len(t.encode("utf-8")) for t in par[l]) for l in langs}
        per_split[split] = {
            "n_sentences": n,
            "bytes": b,
            "premium": {l: b[l] / b[ANCHOR] for l in langs},
        }
        for l in langs:
            tot[l] += b[l]
    return {
        "per_split": per_split,
        "bytes_total": tot,
        "premium": {l: tot[l] / tot[ANCHOR] for l in langs},
    }


def fetch_arnett() -> dict | None:
    """Best-effort fetch of published premiums for comparison (not pipeline-critical)."""
    api = f"https://api.github.com/repos/{ARNETT_REPO}/contents/"
    try:
        with urllib.request.urlopen(api, timeout=30) as r:
            listing = json.load(r)
        tables = [e for e in listing if e["name"].endswith((".csv", ".tsv"))]
        for e in tables:
            with urllib.request.urlopen(e["download_url"], timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
            sep = "\t" if e["name"].endswith(".tsv") else ","
            rows = [ln.split(sep) for ln in text.splitlines() if ln.strip()]
            header = [h.strip().lower() for h in rows[0]]
            # find a language-code column and a premium/ratio column
            code_col = next((i for i, h in enumerate(header)
                             if h in ("language", "lang", "iso", "code", "language_code")), 0)
            val_cols = [i for i, h in enumerate(header) if "premium" in h or "ratio" in h]
            if not val_cols:
                continue
            found = {}
            for row in rows[1:]:
                cell = row[code_col].strip().lower()
                for ours, iso in ARNETT_CODES.items():
                    if cell == iso or cell.startswith(iso + "_"):
                        try:
                            found.setdefault(ours, float(row[val_cols[0]]))
                        except (ValueError, IndexError):
                            pass
            if len(found) >= 4:
                return {"source_file": e["name"], "premium": found}
    except Exception as exc:  # network/GitHub issues must not block calibration
        print(f"[byte_premium] Arnett tool fetch failed ({exc}); skipping comparison")
    return None


def run(out_dir: Path | None = None) -> dict:
    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "byte_premium")
    res = compute()
    res["arnett_comparison"] = fetch_arnett()
    with open(out_dir / "byte_premiums.json", "w") as f:
        json.dump(res, f, indent=2)

    lines = ["| lang | premium (dev) | premium (devtest) | premium (combined) | Arnett et al. |",
             "|---|---|---|---|---|"]
    arn = (res["arnett_comparison"] or {}).get("premium", {})
    for l in LANGS:
        lines.append(
            f"| {l} | {res['per_split']['dev']['premium'][l]:.4f} "
            f"| {res['per_split']['devtest']['premium'][l]:.4f} "
            f"| {res['premium'][l]:.4f} "
            f"| {arn.get(l, float('nan')):.4f} |")
    (out_dir / "byte_premiums.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def load_premiums() -> dict[str, float]:
    path = RESULTS / "byte_premium" / "byte_premiums.json"
    if not path.exists():
        raise FileNotFoundError("run `xscript byte-premium` first")
    return json.loads(path.read_text())["premium"]
