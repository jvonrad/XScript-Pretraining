#!/usr/bin/env python
"""Upload native-stack runs (e.g. the `en-X__unigram_20lang` bilinguals) to
jvonrad/xscript-eval in the repo's established layout.

For each run:
  * ROSTER entries mirror the existing bilingual families
    (`en-X-fair`, `-2b`, `-5b`, `-10b`, `-15b`, `-23b`): the cooled final plus
    the snapshot nearest each reference budget, as
    `runs/<friendly>/checkpoints/final.pt.part000..N` + `n_parts.txt`
    (900 MB chunks, what every other dir uses), with a `models.json` entry
    {orig_run, tok, langs} appended in place (models.json is NOT sorted;
    indent 2, no trailing newline -- re-serialising any other way turns a
    6-entry addition into a 100+-entry diff).
  * `--archive-all` additionally uploads every OTHER snapshot, `last.pt` and
    the per-rank ZeRO optimizer shards whole (no chunking; we are on a compute
    box) under `runs/<orig_run>/checkpoints/`, so nothing on this ephemeral
    scratch is lost. No models.json entries for those.
  * the tokenizer dir (`tokenizers/<tok>/{meta.json,sp.model}`).

Idempotent: any remote file already present at the right size is skipped.

    HF_HUB_ENABLE_HF_TRANSFER=1 python upload_native_runs.py --repo jvonrad/xscript-eval \\
        --runs en-de__unigram_20lang ... --suffix 20lang --tok unigram_20lang --archive-all
"""
import argparse, json, math, os, re, sys, time
from pathlib import Path

CHUNK = 900 * 1024 * 1024
# reference budgets of the existing bilingual rosters (tokens), from models.json
BUDGETS = {"2b": 1.753e9, "5b": 4.753e9, "10b": 9.753e9, "15b": 14.755e9, "23b": 22.757e9}


def snapshots(ckdir: Path):
    out = {}
    for p in ckdir.glob("step*_*M.pt"):
        m = re.match(r"step(\d+)_(\d+)M\.pt", p.name)
        out[p.name] = int(m.group(2)) * 1e6
    return out


def split(src: Path, out_dir: Path):
    n = math.ceil(src.stat().st_size / CHUNK)
    existing = sorted(out_dir.glob("final.pt.part*")) if out_dir.exists() else []
    if len(existing) == n and sum(p.stat().st_size for p in existing) == src.stat().st_size:
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        p.unlink()
    parts = []
    with open(src, "rb") as r:
        for i in range(n):
            dst = out_dir / f"final.pt.part{i:03d}"; left = CHUNK
            with open(dst, "wb") as w:
                while left > 0:
                    buf = r.read(min(64 << 20, left))
                    if not buf:
                        break
                    w.write(buf); left -= len(buf)
            parts.append(dst)
    assert sum(p.stat().st_size for p in parts) == src.stat().st_size
    (out_dir / "n_parts.txt").write_text(str(n))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--suffix", required=True, help="friendly-name condition suffix, e.g. 20lang")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--scratch", default=os.environ.get("XSCRIPT_SCRATCH", "/mnt/scratch/xscript"))
    ap.add_argument("--staging", default="/mnt/scratch/_hf_parts")
    ap.add_argument("--archive-all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    sizes = {s.rfilename: s.size for s in api.repo_info(a.repo, files_metadata=True).siblings}

    def present(key, path):
        return key in sizes and sizes[key] == path.stat().st_size

    def up_folder(local: Path, remote_dir: str, names, msg):
        todo = [n for n in names if not present(f"{remote_dir}/{n}", local / n)]
        if not todo:
            print(f"   all {len(names)} files already present in {remote_dir}"); return
        gb = sum((local / n).stat().st_size for n in todo) / 1e9
        print(f"   uploading {len(todo)} files ({gb:.1f} GB) -> {remote_dir}", flush=True)
        if a.dry_run:
            return
        t0 = time.time()
        api.upload_folder(folder_path=str(local), path_in_repo=remote_dir, repo_id=a.repo,
                          allow_patterns=todo, commit_message=msg)
        print(f"   done in {(time.time()-t0)/60:.1f} min ({gb/(time.time()-t0)*1e3:.0f} MB/s)", flush=True)

    mj_path = Path(api.hf_hub_download(a.repo, "models.json", force_download=True))
    mj = json.loads(mj_path.read_text()); added = []
    for run in a.runs:
        ckdir = Path(a.scratch) / "runs" / run / "checkpoints"
        pair = run.split("__")[0]; langs = pair.split("-")
        friendly = f"{pair}-{a.suffix}"
        snaps = snapshots(ckdir)
        roster = [(friendly, "final.pt")]
        for tag, tgt in BUDGETS.items():
            name = min(snaps, key=lambda n: abs(snaps[n] - tgt))
            if abs(snaps[name] - tgt) > 0.05e9:
                sys.exit(f"{run}: no snapshot within 50M of {tag} ({tgt/1e9:.3f}B); nearest {name}")
            roster.append((f"{friendly}-{tag}", name))
        print(f"\n=== {run} -> {friendly}  ({len(snaps)} snapshots)")
        for fr, fname in roster:
            print(f"  {fr:18s} <- {fname}")
            parts = [] if a.dry_run else split(ckdir / fname, Path(a.staging) / fr)
            if not a.dry_run:
                up_folder(Path(a.staging) / fr, f"runs/{fr}/checkpoints",
                          [p.name for p in parts] + ["n_parts.txt"], f"{fr}: checkpoint")
            if fr not in mj:
                mj[fr] = {"orig_run": run if fname == "final.pt" else f"{run}__{fname[:-3]}",
                          "tok": a.tok, "langs": langs}
                added.append(fr)
        if a.archive_all:
            rest = sorted(n for n in snaps if n not in {f for _, f in roster})
            rest += [p.name for p in ckdir.glob("last.pt")] + sorted(p.name for p in ckdir.glob("last.optim.rank*.pt"))
            print(f"  archive: {len(rest)} files whole -> runs/{run}/checkpoints/")
            up_folder(ckdir, f"runs/{run}/checkpoints", rest, f"{run}: full snapshot archive")
    tokdir = Path(a.scratch) / "tokenizers" / a.tok
    print(f"\n=== tokenizer {a.tok}")
    up_folder(tokdir, f"tokenizers/{a.tok}", ["meta.json", "sp.model"], f"tokenizer {a.tok}")
    if added and not a.dry_run:
        mj_path.write_text(json.dumps(mj, indent=2))   # same serialisation as the existing file
        api.upload_file(path_or_fileobj=str(mj_path), path_in_repo="models.json", repo_id=a.repo,
                        commit_message=f"models.json: +{len(added)} ({', '.join(added)})")
        print(f"\nmodels.json: {len(mj)} entries (+{len(added)})")
    else:
        print(f"\nmodels.json would gain {len(added)} entries: {added}")
    print("UPLOAD_DONE")


if __name__ == "__main__":
    main()
