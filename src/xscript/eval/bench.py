"""Downstream multilingual benchmarks via lm-evaluation-harness.

Complements the intrinsic BPB eval (eval/bpb.py) and the bespoke
cross-lingual-transfer/representation analyses (eval/bts.py, eval/alignment.py)
with real task accuracy: Global-MMLU (knowledge), Belebele (reading
comprehension), XNLI (natural language inference) -- all covering en/de/fr/ar/zh.

By default, evaluation is restricted to the languages in the checkpoint's
training mixture: three tasks for a monolingual run and six for a bilingual
run. ``--tasks`` remains an explicit override for targeted/OOD analyses.

`XScriptLM` wraps our own Transformer/Tok (not HF-standard) in lm_eval's `LM`
interface. Scoring follows the exact shifted-LM convention already used in
eval/bpb.py's score_texts: model(x, y) with x=seq[:-1], y=seq[1:] returns
logits where logits[:, j] predicts y[j].
"""
import importlib.metadata
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..tok.wrapper import BOS_ID, EOS_ID, PAD_ID

DEFAULT_TASKS = {
    "global_mmlu": ["global_mmlu_en", "global_mmlu_de", "global_mmlu_fr",
                    "global_mmlu_ar", "global_mmlu_zh"],
    "belebele": ["belebele_eng_Latn", "belebele_deu_Latn", "belebele_fra_Latn",
                 "belebele_arb_Arab", "belebele_zho_Hans"],
    "xnli": ["xnli_en", "xnli_de", "xnli_fr", "xnli_ar", "xnli_zh"],
}

LANG_ORDER = ("en", "de", "fr", "ar", "zh")
TASKS_BY_LANG = {
    lang: [DEFAULT_TASKS[family][i] for family in DEFAULT_TASKS]
    for i, lang in enumerate(LANG_ORDER)
}


def tasks_for_langs(langs: list[str]) -> list[str]:
    """Harness task names for exactly the languages in a run's mixture."""
    unknown = [lang for lang in langs if lang not in TASKS_BY_LANG]
    if unknown:
        raise ValueError(f"no downstream task mapping for languages: {unknown}")
    # Keep benchmark families together in the output, then run language order.
    return [TASKS_BY_LANG[lang][family_i]
            for family_i in range(len(DEFAULT_TASKS)) for lang in langs]


class XScriptLM:
    """lm_eval.api.model.TemplateLM subclass wrapping our Transformer + Tok.

    Inherits from TemplateLM lazily (import-time, so lm_eval/torch stay
    optional deps of the base package) via _make_lm() below.
    """

    def __init__(self, model, tok, device, max_seq_len: int, batch_size: int = 4):
        super().__init__()
        self.model = model.eval()
        self.tok = tok
        # lm_eval.api.model.LM exposes `device` as a read-only property backed
        # by `_device`; assigning self.device raises AttributeError.
        self._device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokenizer = None  # no chat template support needed for these tasks

    @property
    def eot_token_id(self) -> int:
        return EOS_ID

    @property
    def prefix_token_id(self) -> int:
        # our documents are always BOS-prefixed, not EOS-prefixed
        return BOS_ID

    def tok_encode(self, string: str, add_special_tokens=None, **kwargs) -> list[int]:
        return self.tok.encode(string, bos=False, eos=False)

    def _prepare(self, context_enc: list[int], continuation_enc: list[int]) -> list[int]:
        """Return a model-ready sequence with one BOS and an intact target.

        TemplateLM supplies ``[prefix_token_id]`` for an empty string context,
        whereas non-empty contexts contain no special token.  Normalize both
        cases here so BOS is added exactly once.  Context is left-truncated;
        benchmark answer continuations are never silently truncated.
        """
        if not continuation_enc:
            return []
        has_bos = bool(context_enc) and context_enc[0] == BOS_ID
        context = context_enc[1:] if has_bos else context_enc
        if len(continuation_enc) > self.max_seq_len:
            raise ValueError(
                f"continuation has {len(continuation_enc)} tokens, exceeding "
                f"max_seq_len={self.max_seq_len}"
            )
        budget = self.max_seq_len - len(continuation_enc)
        context = context[-budget:] if budget < len(context) else context
        return [BOS_ID] + context + continuation_enc

    @torch.no_grad()
    def _score_batch(self, batch) -> list[tuple[float, bool]]:
        """Score variable-length requests with right padding.

        Padding is strictly after each real sequence, so causal attention
        cannot let it affect any scored position.  Passing targets asks our
        Transformer for all-position logits; the returned scalar loss is
        intentionally ignored.
        """
        prepared = [(self._prepare(list(c), list(k)), len(k)) for c, k in batch]
        out: list[tuple[float, bool] | None] = [None] * len(prepared)
        active = [(i, seq, n) for i, (seq, n) in enumerate(prepared) if n]
        for i, (_, n) in enumerate(prepared):
            if not n:
                out[i] = (0.0, True)
        if not active:
            return out  # type: ignore[return-value]

        width = max(len(seq) - 1 for _, seq, _ in active)
        x = torch.full((len(active), width), PAD_ID, dtype=torch.long,
                       device=self.device)
        y = torch.full((len(active), width), -100, dtype=torch.long,
                       device=self.device)
        lengths = []
        for row, (_, seq, _) in enumerate(active):
            m = len(seq) - 1
            lengths.append(m)
            x[row, :m] = torch.tensor(seq[:-1], device=self.device)
            y[row, :m] = torch.tensor(seq[1:], device=self.device)

        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if self.device.type == "cuda" else _null())
        with amp:
            logits, _ = self.model(x, y)
        for row, (out_i, _, n) in enumerate(active):
            m = lengths[row]
            cont_logits = logits[row, m - n:m, :].float()
            target = y[row, m - n:m]
            logprobs = F.log_softmax(cont_logits, dim=-1)
            token_lp = logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
            greedy = bool((cont_logits.argmax(-1) == target).all().item())
            out[out_i] = (float(token_lp.sum().item()), greedy)
        return out  # type: ignore[return-value]

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        from tqdm import tqdm
        out = []
        batches = range(0, len(requests), self.batch_size)
        for st in tqdm(batches, disable=disable_tqdm, desc="[bench] scoring"):
            chunk = requests[st:st + self.batch_size]
            out.extend(self._score_batch([(c, k) for _, c, k in chunk]))
        return out

    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        from lm_eval import utils
        from tqdm import tqdm
        out = []
        for req in tqdm(requests, disable=disable_tqdm, desc="[bench] rolling"):
            (text,) = req.args
            ids = self.tok_encode(text)
            windows = list(utils.get_rolling_token_windows(
                token_list=ids, prefix_token=BOS_ID,
                max_seq_len=self.max_seq_len, context_len=1,
            ))
            # The utility's contexts are already complete windows (the first
            # starts with BOS), so score them without _prepare adding BOS.
            total = 0.0
            for context, target in windows:
                x = torch.tensor(context, device=self.device).unsqueeze(0)
                # Input and prediction windows are aligned by the harness;
                # only the final len(target) logits are part of this window.
                y_ids = [-100] * (len(context) - len(target)) + target
                y = torch.tensor(y_ids, device=self.device).unsqueeze(0)
                logits, _ = self.model(x, y)
                n = len(target)
                lp = F.log_softmax(logits[0, -n:, :].float(), -1)
                total += float(lp.gather(1, y[0, -n:].unsqueeze(1)).sum().item())
            out.append(total)
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False):
        from tqdm import tqdm
        out = []
        for req in tqdm(requests, disable=disable_tqdm, desc="[bench] generating"):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            max_gen = (gen_kwargs.get("max_gen_toks", 256)
                      if isinstance(gen_kwargs, dict) else 256)
            ids = [BOS_ID] + self.tok_encode(context)[-(self.max_seq_len - 1):]
            gen = []
            text_so_far = ""
            for _ in range(max_gen):
                x = torch.tensor(ids[-self.max_seq_len:], device=self.device).unsqueeze(0)
                logits = self.model(x)                     # (1, 1, vocab) -- no targets
                next_id = int(logits[0, -1].argmax(-1).item())
                if next_id == EOS_ID:
                    break
                gen.append(next_id)
                ids.append(next_id)
                text_so_far = self.tok.decode(gen)
                if until and any(u in text_so_far for u in until):
                    for u in until:
                        idx = text_so_far.find(u)
                        if idx != -1:
                            text_so_far = text_so_far[:idx]
                    break
            out.append(text_so_far)
        return out


def _make_lm(model, tok, device, max_seq_len):
    """Bind XScriptLM to lm_eval.api.model.TemplateLM at call time (keeps
    lm_eval/torch optional for anything that only imports xscript.eval.bench
    for DEFAULT_TASKS)."""
    from lm_eval.api.model import TemplateLM

    class _Bound(XScriptLM, TemplateLM):
        def __init__(self):
            XScriptLM.__init__(self, model, tok, device, max_seq_len)

    return _Bound()


def run(run_name: str, tok_name: str, tag: str = "final", tasks: list[str] | None = None,
        num_fewshot: int = 0, limit: int | float | None = None,
        out_dir: Path | None = None, log_wandb: bool = True,
        batch_size: int = 4) -> dict:
    """Evaluate a checkpoint on its training languages by default."""
    import lm_eval
    from ..model import ModelConfig, Transformer
    from ..tok.wrapper import Tok
    from ..paths import RUNS, RESULTS, tokenizer_dir, ensure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(RUNS / run_name / "checkpoints" / f"{tag}.pt",
                    map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
    model.load_state_dict(ck["model"])
    tok = Tok(tokenizer_dir(tok_name))
    if tok_name != ck["cfg"]["tok_name"]:
        raise ValueError(f"checkpoint uses {ck['cfg']['tok_name']}, not {tok_name}")

    task_list = tasks if tasks is not None else tasks_for_langs(ck["cfg"]["langs"])
    adapter = _make_lm(model, tok, device, model.cfg.max_seq_len)
    adapter.batch_size = batch_size

    results = lm_eval.simple_evaluate(
        model=adapter, tasks=task_list, num_fewshot=num_fewshot,
        batch_size=1, limit=limit, log_samples=False, confirm_run_unsafe_code=True,
    )
    def _accuracy(rec):
        # Use ordinary accuracy consistently across all three benchmark
        # families. Belebele additionally reports length-normalized accuracy,
        # which remains available in the preserved raw harness output.
        return rec.get("acc,none", rec.get("acc"))

    scores = {}
    groups = results.get("groups", {})
    subtasks = results.get("results", {})
    for name in task_list:
        rec = groups.get(name, subtasks.get(name, {}))
        scores[name] = _accuracy(rec)

    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "bench")
    payload = {
        "run": run_name, "checkpoint": tag, "tokenizer": tok_name,
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "num_fewshot": num_fewshot, "limit": limit, "tasks": task_list,
        "scores": scores, "results": results.get("results", {}),
        "groups": groups, "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
    }
    (out_dir / f"{run_name}_{tag}.json").write_text(
        json.dumps(payload, indent=2, default=_json_default)
    )
    print(f"[bench] {run_name} ({tag}): " +
          ", ".join(f"{k}={v:.4f}" for k, v in scores.items() if v is not None))

    if log_wandb:
        try:
            import wandb
            wb = wandb.init(project="XScript-Pretraining", id=run_name, resume="allow")
            wb.log({f"bench/{k}": v for k, v in scores.items() if v is not None})
            wb.finish()
        except Exception as exc:
            print(f"[bench] wandb logging skipped ({exc})")

    return scores


class _null:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def _json_default(value):
    """Serialize NumPy scalars and other scalar-like harness values."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
