"""Upload the two finished Chinese 15B checkpoints to jvonrad/xscript-eval.

Matches the repo's existing convention exactly:
  runs/<name>/checkpoints/final.pt.part000..004  (900 MiB chunks, last = remainder)
  runs/<name>/checkpoints/n_parts.txt            ("5", no trailing newline)
and adds the two models.json entries.

Uses the 14.756B snapshot (step15865_14756M) rather than the 15.0B final, because
every existing "-15b" model in this repo is ~14.754B tokens -- this keeps the new
Chinese models exactly budget-matched to the ar/de/en/fr -15b series.
"""
import json
import math
import os

from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]
REPO = "jvonrad/xscript-eval"
CHUNK = 900 * 1024 * 1024  # 900 MiB, matching existing parts
SRC_CKPT = "step15865_14756M.pt"
STAGE = "/mnt/scratch/hf_upload"

JOBS = [
    # (local run dir,            hf name,          tokenizer)
    ("zh__unigram_destarved", "zh-fair-15b", "unigram_destarved"),
    ("zh__unigram_starved", "zh-starved-15b", "unigram_starved"),
]

api = HfApi(token=TOKEN)

for run, name, tok in JOBS:
    src = f"/mnt/scratch/xscript/runs/{run}/checkpoints/{SRC_CKPT}"
    size = os.path.getsize(src)
    nparts = math.ceil(size / CHUNK)
    outdir = os.path.join(STAGE, name)
    os.makedirs(outdir, exist_ok=True)
    print(f"[{name}] source {src} ({size:,} bytes) -> {nparts} parts", flush=True)

    with open(src, "rb") as f:
        for i in range(nparts):
            part = os.path.join(outdir, f"final.pt.part{i:03d}")
            data = f.read(CHUNK)
            with open(part, "wb") as g:
                g.write(data)
            print(f"[{name}]   wrote part{i:03d} ({len(data):,} bytes)", flush=True)
    with open(os.path.join(outdir, "n_parts.txt"), "w") as f:
        f.write(str(nparts))

    for i in range(nparts):
        part = os.path.join(outdir, f"final.pt.part{i:03d}")
        api.upload_file(path_or_fileobj=part,
                        path_in_repo=f"runs/{name}/checkpoints/final.pt.part{i:03d}",
                        repo_id=REPO, repo_type="model")
        print(f"[{name}]   uploaded part{i:03d}", flush=True)
    api.upload_file(path_or_fileobj=os.path.join(outdir, "n_parts.txt"),
                    path_in_repo=f"runs/{name}/checkpoints/n_parts.txt",
                    repo_id=REPO, repo_type="model")
    print(f"[{name}] uploaded n_parts.txt -- checkpoint DONE", flush=True)

# ---- models.json ----
mj = hf_hub_download(REPO, "models.json", token=TOKEN)
d = json.load(open(mj))
for run, name, tok in JOBS:
    d[name] = {"orig_run": f"{run}__{SRC_CKPT[:-3]}", "tok": tok, "langs": ["zh"]}
    print(f"models.json += {name}: {json.dumps(d[name])}", flush=True)
local_mj = os.path.join(STAGE, "models.json")
with open(local_mj, "w") as f:
    json.dump(d, f, indent=2)
api.upload_file(path_or_fileobj=local_mj, path_in_repo="models.json",
                repo_id=REPO, repo_type="model")
print(f"models.json uploaded ({len(d)} entries)", flush=True)
print("UPLOAD_ALL_DONE", flush=True)
