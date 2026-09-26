"""
PPL + Semantic Surprise (Beam) — single-poem mode, tables only.

For each word position:
- surprisal_bits  -log2 P(word | context)   (bits)
- cos_dist_centroid   uniform (unweighted) mean cosine distance between actual
                  word embedding and the top-k predicted word embeddings

Surprisal captures structural/morphological expectation (how likely).
cos_dist_centroid captures the semantic field (how semantically close), independent
of probability — giving a clean separation between the two axes.

Beam search (beam_width=2) expands each word-start token into multiple
candidate words. Nucleus: keep collecting complete-word candidates until
PROB_MASS_THRESHOLD of probability mass is covered (or MAX_PREDICTIONS reached).

Optimized for runtime WITHOUT changing the measurement itself.
Plot generation removed; CSV tables only.
"""

import copy
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from typing import Any, cast

import pandas as pd
from clearml import Task
from dotenv import load_dotenv


# ===============================
# LEMMA UTILS (DictaBERT-lex)
# ===============================
def _load_dictabert_lex():
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-lex")
    model = AutoModel.from_pretrained(
        "dicta-il/dictabert-lex", trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


try:
    _lex_tokenizer, _lex_model = _load_dictabert_lex()
    print("DictaBERT-lex loaded.", flush=True)
except Exception as e:
    _lex_tokenizer, _lex_model = None, None
    print(f"WARNING: DictaBERT-lex not available ({e}). Lemmatization disabled.", flush=True)

_lemma_cache: dict[str, str] = {}


def get_lemma_batch(words: list[str]) -> list[str]:
    """Lemmatize a list of words in one batch call. Results are cached."""
    if _lex_model is None:
        return list(words)

    to_lookup = [w for w in words if w not in _lemma_cache]
    if to_lookup:
        try:
            preds = _lex_model.predict(to_lookup, _lex_tokenizer)
            for word, pred in zip(to_lookup, preds):
                if pred and len(pred) > 0:
                    lemma = pred[0][1] if isinstance(pred[0], tuple) else str(pred[0])
                    _lemma_cache[word] = lemma if lemma and lemma != "[BLANK]" else word
                else:
                    _lemma_cache[word] = word
        except Exception:
            for w in to_lookup:
                _lemma_cache[w] = w

    return [_lemma_cache.get(w, w) for w in words]


def get_lemma(word: str) -> str:
    return get_lemma_batch([word])[0]


# ===============================
# CONSTANTS
# ===============================
MODEL_NAME = "dicta-il/DictaLM-3.0-1.7B-Base"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Data")
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Outputs", "semantic_surprise"
)

# Which poem to process (0 = first alphabetically, 1 = second, …)
POEM_INDEX = 0

# "remote"  → register + execute_remotely (RunPod agent)
# "clearml" → run locally, report to ClearML
# "local"   → run locally, no ClearML at all
RUN_MODE = "remote"

HEBREW_COMBINING_MARKS = re.compile(
    r"[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]"
)
CLEAN_WORD_RE = re.compile(r"[^\u05D0-\u05EAa-zA-Z]")

PROB_MASS_THRESHOLD = 0.65
MIN_PREDICTIONS = 5
MAX_PREDICTIONS = 50
MAX_WORD_TOKENS = 8
BEAM_WIDTH = 2
SCAN_K = 500  # כמה טוקנים לסרוק בחיפוש אסימוני-תחילת-מילה (זהה לפייפליין האנגלי)

def _no_beam() -> bool:
    return os.getenv("PPL_HE_NO_BEAM", "false").strip().lower() in {"1", "true", "yes"}

def _use_weighted() -> bool:
    """v4 = weighted (default); v3 = unweighted (PPL_HE_WEIGHTED=false)."""
    return os.getenv("PPL_HE_WEIGHTED", "true").strip().lower() not in {"0", "false", "no"}
WORD_START_PREFIX = "\u0120"  # Ġ — BPE word-start marker
_SOFIT = set("םןףךץ")  # Hebrew final letters — always end a word


def strip_niqqud(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = HEBREW_COMBINING_MARKS.sub("", text)
    return unicodedata.normalize("NFKC", text)


def split_to_words(text: str) -> list[str]:
    # Split on whitespace, then split hyphenated tokens (e.g. "את-נשמת" → "את", "נשמת")
    words = []
    for w in text.split():
        if "-" in w:
            words.extend(part for part in w.split("-") if part)
        else:
            words.append(w)
    return [w for w in words if w.strip()]


def clean_word(word: str) -> str:
    # Efficient clean_word using precompiled regex
    return CLEAN_WORD_RE.sub("", word)


# ========== BATCH HELPERS ========== #
def _clone_cache_batch(self, caches):
    """Clone a batch of caches efficiently."""
    return [self._clone_cache(c) for c in caches]


def _advance_selected_beams_batched(self, beams, device):
    """Advance a batch of beams (token_ids, cache, logits) by one token in parallel."""
    torch = self._torch
    # beams: list of (token_ids, prob, cache, logits, next_token_id)
    if not beams:
        return []
    next_token_ids = [b[4] for b in beams]
    caches = [b[2] for b in beams]
    # Clone caches for each beam
    caches = self._clone_cache_batch(caches)
    # Prepare input tensor
    inp = torch.tensor([[tid] for tid in next_token_ids], device=device)
    # Forward all in batch
    logits_list = []
    new_caches = []
    for i in range(len(beams)):
        logits, new_cache = self._forward(inp[i : i + 1], past_key_values=caches[i])
        logits_list.append(logits)
        new_caches.append(new_cache)
    return logits_list, new_caches


class SemanticSurpriseAnalyser:
    def __init__(self, model_name: str = MODEL_NAME):
        import torch
        import torch.nn.functional as F  # noqa: N812
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._F = F

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = self.device.type == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32

        print(f"Loading model {model_name} on {self.device} ({dtype})...", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("  Loading weights...", flush=True)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        print("  Extracting embeddings...", flush=True)
        embed_layer = self.model.get_input_embeddings()
        self.embeddings = embed_layer.weight.detach().float()  # type: ignore[union-attr]

        print("  Building word-start index...", flush=True)
        vocab = self.tokenizer.get_vocab()
        self._word_start_ids: set[int] = {
            tid for token, tid in vocab.items() if token.startswith(WORD_START_PREFIX)
        }

        print(
            f"  Ready. "
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params, "
            f"embed_dim={self.embeddings.shape[1]}, "
            f"word-start tokens={len(self._word_start_ids)}"
        )

    def _is_word_start(self, token_id: int) -> bool:
        return token_id in self._word_start_ids

    def _word_embedding(self, token_ids: list[int]):
        return self.embeddings[token_ids].mean(dim=0)

    def _tokenize_word(self, word: str) -> list[int]:
        return self.tokenizer.encode(" " + word, add_special_tokens=False)

    def _lemmatize_word(self, word: str) -> str:
        return get_lemma(word)

    def _forward(self, input_ids, past_key_values=None):
        t0 = time.time()
        out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if hasattr(self, "_prof"):
            self._prof["forward"]["count"] += 1
            self._prof["forward"]["time"] += time.time() - t0
        return out.logits, out.past_key_values

    def _clone_cache(self, cache):
        t0 = time.time()
        try:
            from transformers.cache_utils import DynamicCache

            src = cast(Any, cache)
            new = DynamicCache()
            dst = cast(Any, new)

            for k, v in zip(src.key_cache, src.value_cache):
                dst.key_cache.append(k.clone())
                dst.value_cache.append(v.clone())

            result = new
        except (AttributeError, ImportError):
            result = copy.deepcopy(cache)
        if hasattr(self, "_prof"):
            self._prof["clone_cache"]["count"] += 1
            self._prof["clone_cache"]["time"] += time.time() - t0
        return result

    def _reset_profiling(self):
        self._prof = {
            "clone_cache": {"count": 0, "time": 0.0},
            "forward": {"count": 0, "time": 0.0},
            "expand_to_words": {"count": 0, "time": 0.0},
            "word_surprisal": {"count": 0, "time": 0.0},
            "cosine_distances": {"count": 0, "time": 0.0},
            "lemmatize": {"count": 0, "time": 0.0},
        }

    def _print_profiling(self, label: str = ""):
        p = self._prof
        total = sum(v["time"] for v in p.values())
        print(f"\n  === PROFILING {label} ===")
        for name, v in sorted(p.items(), key=lambda x: -x[1]["time"]):
            pct = (v["time"] / total * 100) if total > 0 else 0
            print(
                f"    {name:25s}: {v['count']:5d} calls, {v['time']:7.2f}s ({pct:5.1f}%)"
            )
        print(f"    {'TOTAL':25s}: {total:7.2f}s")

    def _init_cache(self, first_word: str):
        torch = self._torch
        tokens = self.tokenizer.encode(first_word + " ", add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)

        with torch.inference_mode():
            logits, cache = self._forward(input_ids)

        return cache, logits[0, -1, :]

    def _advance_cache(self, cache, actual_word: str):
        torch = self._torch
        tokens = self.tokenizer.encode(
            " " + actual_word + " ", add_special_tokens=False
        )

        logits = None
        with torch.inference_mode():
            for t in tokens:
                inp = torch.tensor([[t]], device=self.device)
                logits, cache = self._forward(inp, past_key_values=cache)

        assert logits is not None, f"No tokens for word: {actual_word!r}"
        return cache, logits[0, -1, :]

    def _analyze_position(
        self,
        ctx_cache,
        next_logits,
        actual_word: str,
        prob_threshold: float = PROB_MASS_THRESHOLD,
        min_preds: int = MIN_PREDICTIONS,
        max_preds: int = MAX_PREDICTIONS,
    ) -> dict:
        torch = self._torch
        F = self._F

        next_log_probs = F.log_softmax(next_logits.float(), dim=0)

        actual_tokens = self._tokenize_word(actual_word)
        surprisal = self._word_surprisal(ctx_cache, actual_tokens, next_log_probs)

        _, top_ids = torch.topk(next_logits.float(), SCAN_K)

        predictions: list[tuple[str, float, list[int]]] = []
        seen: set[str] = set()
        cumulative_prob = 0.0

        if _no_beam():
            # Simple mode: take word-start tokens directly, no beam expansion
            for tid in top_ids:
                if len(predictions) >= max_preds:
                    break
                if cumulative_prob >= prob_threshold and len(predictions) >= min_preds:
                    break
                tid_int = int(tid.item())
                if not self._is_word_start(tid_int):
                    continue
                raw = self.tokenizer.decode([tid_int]).strip()
                word = clean_word(raw)
                if not word or word in seen:
                    continue
                prob = next_log_probs[tid_int].exp().item()
                seen.add(word)
                predictions.append((word, prob, [tid_int]))
                cumulative_prob += prob
            lemma_counter: Counter[str] = Counter()
            lemma_forms: dict[str, list[str]] = {}
        else:
            _n_word_starts = 0
            _n_expand_calls = 0
            _last_scan_idx = 0

            for scan_idx, tid in enumerate(top_ids):
                if len(predictions) >= max_preds:
                    break
                if cumulative_prob >= prob_threshold and len(predictions) >= min_preds:
                    break

                tid_int = int(tid.item())
                if not self._is_word_start(tid_int):
                    continue

                _n_word_starts += 1
                first_prob = next_log_probs[tid_int].exp().item()

                _n_expand_calls += 1
                expanded_words = self._expand_to_words(
                    ctx_cache,
                    tid_int,
                    first_prob,
                    beam_width=BEAM_WIDTH,
                )
                for word, word_prob, tids in expanded_words:
                    if word not in seen:
                        seen.add(word)
                        predictions.append((word, word_prob, tids))
                        cumulative_prob += word_prob
                        if len(predictions) >= max_preds:
                            break
                        if cumulative_prob >= prob_threshold and len(predictions) >= min_preds:
                            break
                _last_scan_idx = scan_idx

            predictions.sort(key=lambda x: x[1], reverse=True)

            t_lemma = time.time()
            # Batch lemmatize all predictions
            pred_words = [w for w, _, _ in predictions]
            lemmas = get_lemma_batch(pred_words)

            # Group by lemma: sum probs, keep highest-prob form as representative
            lemma_groups: dict[str, list[tuple]] = {}
            for (word, prob, tids), lemma in zip(predictions, lemmas):
                key = lemma if lemma else word
                lemma_groups.setdefault(key, []).append((word, prob, tids))

            # Build deduplicated predictions
            dedup: list[tuple[str, float, list[int]]] = []
            for lemma_key, group in lemma_groups.items():
                total_prob = sum(p for _, p, _ in group)
                repr_word, _, repr_tids = max(group, key=lambda x: x[1])
                dedup.append((repr_word, total_prob, repr_tids))
            dedup.sort(key=lambda x: x[1], reverse=True)

            # Logging
            lemma_counter = Counter({k: len(v) for k, v in lemma_groups.items()})
            lemma_forms = {k: [w for w, _, _ in v] for k, v in lemma_groups.items()}

            if hasattr(self, "_prof"):
                self._prof["lemmatize"]["count"] += 1
                self._prof["lemmatize"]["time"] += time.time() - t_lemma

            predictions = dedup

        cos = self._cosine_distances(actual_word, predictions)

        return {
            "surprisal_bits": surprisal,
            "predictions": predictions,
            "prob_mass": cumulative_prob,
            "lemma_counts": dict(lemma_counter),
            "lemma_forms": lemma_forms,
            **cos,
        }

    def _word_surprisal(
        self, ctx_cache, word_tokens: list[int], first_log_probs
    ) -> float:
        t0 = time.time()
        torch = self._torch
        F = self._F

        if not word_tokens:
            if hasattr(self, "_prof"):
                self._prof["word_surprisal"]["count"] += 1
                self._prof["word_surprisal"]["time"] += time.time() - t0
            return float("inf")

        total_log_prob = first_log_probs[word_tokens[0]].item()

        if len(word_tokens) > 1:
            cache = self._clone_cache(ctx_cache)
            with torch.inference_mode():
                inp = torch.tensor([[word_tokens[0]]], device=self.device)
                logits, cache = self._forward(inp, past_key_values=cache)

                for t in word_tokens[1:]:
                    log_probs = F.log_softmax(logits[0, -1, :].float(), dim=0)
                    total_log_prob += log_probs[t].item()
                    inp = torch.tensor([[t]], device=self.device)
                    logits, cache = self._forward(inp, past_key_values=cache)

        result = -total_log_prob / math.log(2)
        if hasattr(self, "_prof"):
            self._prof["word_surprisal"]["count"] += 1
            self._prof["word_surprisal"]["time"] += time.time() - t0
        return result

    def _expand_to_words(
        self,
        ctx_cache,
        first_token_id: int,
        first_prob: float,
        beam_width: int = BEAM_WIDTH,
    ) -> list[tuple[str, float, list[int]]]:
        """
        Batched beam expansion for word candidates. Measurement unchanged.
        """
        t0 = time.time()
        torch = self._torch
        device = self.device

        cache0 = self._clone_cache(ctx_cache)
        inp = torch.tensor([[first_token_id]], device=device)

        with torch.inference_mode():
            logits0, cache0 = self._forward(inp, past_key_values=cache0)

        # beam = (token_ids, joint_prob, cache, logits)
        beams = [([first_token_id], first_prob, cache0, logits0)]
        finished: list[tuple[list[int], float]] = []

        with torch.inference_mode():
            for _ in range(MAX_WORD_TOKENS - 1):
                candidates: list[tuple[list[int], float, object, int]] = []

                # Batched candidate expansion
                for toks, prob, b_cache, b_logits in beams:
                    probs = torch.softmax(b_logits[0, -1, :].float(), dim=0)
                    top_ps, top_is = torch.topk(probs, beam_width)

                    beam_finished = False
                    for tp, ti in zip(top_ps, top_is):
                        tid = int(ti.item())
                        if (
                            self._is_word_start(tid)
                            or tid == self.tokenizer.eos_token_id
                        ):
                            if not beam_finished:
                                finished.append((toks, prob))
                                beam_finished = True
                            continue
                        # Early reject: if partial word already has a
                        # Hebrew final letter not at the end, it crossed
                        # a word boundary — no point expanding further.
                        new_toks = toks + [tid]
                        partial = self.tokenizer.decode(
                            new_toks, skip_special_tokens=True
                        ).strip()
                        partial_clean = clean_word(partial)
                        if partial_clean and any(
                            c in _SOFIT for c in partial_clean[:-1]
                        ):
                            continue
                        candidates.append((new_toks, prob * tp.item(), b_cache, tid))

                if not candidates:
                    break

                candidates.sort(key=lambda x: x[1], reverse=True)
                # Batched beam advancement
                new_beams = []
                for toks, prob, parent_cache, last_tid in candidates[:beam_width]:
                    c = self._clone_cache(parent_cache)
                    inp = torch.tensor([[last_tid]], device=device)
                    new_logits, c = self._forward(inp, past_key_values=c)
                    new_beams.append((toks, prob, c, new_logits))
                beams = new_beams

        for toks, prob, _, _ in beams:
            finished.append((toks, prob))

        results: list[tuple[str, float, list[int]]] = []
        seen: set[str] = set()

        for toks, prob in finished:
            raw = self.tokenizer.decode(toks, skip_special_tokens=True).strip()
            tok_strs = [self.tokenizer.decode([t]) for t in toks]
            # Reject if decoded text contains internal whitespace (multiple words)
            if any(c in raw for c in " \t\n"):
                continue
            word = clean_word(raw)
            if not word or word in seen:
                continue
            # Reject mixed-script words (Hebrew + Latin glued together)
            has_heb = any("\u05d0" <= c <= "\u05ea" for c in word)
            has_lat = any(c.isascii() and c.isalpha() for c in word)
            if has_heb and has_lat:
                continue
            # Reject words with a Hebrew final letter (ם ן ף ך ץ) not at the end
            if any(c in _SOFIT for c in word[:-1]):
                continue
            # Validate: re-tokenize and reject if internal word-start tokens
            # are found (means multiple words got glued together)
            re_toks = self.tokenizer.encode(" " + word, add_special_tokens=False)
            if any(self._is_word_start(t) for t in re_toks[1:]):
                continue
            seen.add(word)
            results.append((word, prob, toks))
        if hasattr(self, "_prof"):
            self._prof["expand_to_words"]["count"] += 1
            self._prof["expand_to_words"]["time"] += time.time() - t0
        return results

    def _cosine_distances(
        self, actual_word: str, predictions: list[tuple[str, float, list[int]]]
    ) -> dict:
        t0 = time.time()
        F = self._F
        torch = self._torch

        # In beam mode: lemmatize actual word for a fair semantic comparison
        # (predictions are already deduplicated at lemma level)
        if _no_beam():
            actual_surface = actual_word
        else:
            actual_surface = get_lemma(actual_word) or actual_word

        actual_tokens = self._tokenize_word(actual_surface)
        if not actual_tokens or not predictions:
            return {
                "cos_dist_centroid": float("nan"),
                "cos_dist_min": float("nan"),
            }

        # No-beam mode: use first token only (parallel to English pipeline)
        # Full mode: average all tokens (handles Hebrew multi-token words)
        if _no_beam():
            actual_emb = self._word_embedding(actual_tokens[:1]).unsqueeze(0)
            pred_embs  = torch.stack([self._word_embedding(tids[:1]) for _, _, tids in predictions])
        else:
            # Lemma against lemma. The observed word was lemmatised a few lines
            # above; the candidates are lemmatised here rather than embedded
            # from `tids`, which are the tokens of the group's representative
            # surface form. Embedding one side as a lemma and the other as a
            # surface form scores a distance whenever the model predicted the
            # right word under a different inflection, which is the comparison
            # the lemma deduplication exists to make possible. The paper's
            # corpus was corrected after the fact by recompute_lemma_cosine.py;
            # this file applies the same correction during the measurement so
            # that no post-hoc step is needed here.
            actual_emb = self._word_embedding(actual_tokens).unsqueeze(0)
            pred_lemma_tokens = []
            for word, _, tids in predictions:
                lem = get_lemma(word) or word
                lt = self._tokenize_word(lem)
                pred_lemma_tokens.append(lt if lt else tids)
            pred_embs = torch.stack([self._word_embedding(lt)
                                     for lt in pred_lemma_tokens])

        cos_sims = F.cosine_similarity(actual_emb, pred_embs, dim=1)
        cos_dists = (1.0 - cos_sims).tolist()

        if _use_weighted():
            # v4: probability-weighted mean (Giulianelli et al. 2023)
            probs = self._torch.tensor(
                [p for _, p, _ in predictions], dtype=self._torch.float
            )
            probs = probs / probs.sum()
            centroid_dist = float((probs * self._torch.tensor(cos_dists)).sum())
        else:
            # v3: unweighted mean
            centroid_dist = float(sum(cos_dists) / len(cos_dists))

        result = {
            "cos_dist_centroid": centroid_dist,
            "cos_dist_min": min(cos_dists),
        }
        if hasattr(self, "_prof"):
            self._prof["cosine_distances"]["count"] += 1
            self._prof["cosine_distances"]["time"] += time.time() - t0
        return result

    def analyze(self, text: str) -> list[dict]:
        clean = strip_niqqud(text)
        words = split_to_words(clean)
        if not words:
            return []

        results = []
        total_time = 0.0
        self._reset_profiling()

        rolling_cache, next_logits = self._init_cache(words[0])

        try:
            for i in range(1, len(words)):
                actual_word = words[i]
                actual_clean = clean_word(actual_word)

                t0 = time.time()

                out = self._analyze_position(rolling_cache, next_logits, actual_clean)
                rolling_cache, next_logits = self._advance_cache(
                    rolling_cache, actual_clean
                )

                elapsed = time.time() - t0
                total_time += elapsed

                predictions = out["predictions"]
                pred_words = [w for w, _, _ in predictions]

                try:
                    actual_rank = pred_words.index(actual_clean) + 1
                except ValueError:
                    actual_rank = -1

                prob_coverage = out["prob_mass"]
                n_preds = len(predictions)

                row: dict = {
                    "word_position": i,
                    "actual_word": actual_word,
                    "surprisal_bits": round(out["surprisal_bits"], 4),
                    "cos_dist_centroid": round(out["cos_dist_centroid"], 4),
                    "cos_dist_min": round(out["cos_dist_min"], 4),
                    "prob_mass": round(prob_coverage, 4),
                    "n_predictions": n_preds,
                    "actual_rank": actual_rank,
                    "in_nucleus": actual_rank > 0,
                    "lemma_counts": json.dumps(out["lemma_counts"], ensure_ascii=False),
                    "lemma_forms": json.dumps(out["lemma_forms"], ensure_ascii=False),
                }

                for k in range(n_preds):
                    row[f"pred_{k + 1}"] = predictions[k][0]
                    row[f"prob_{k + 1}"] = round(predictions[k][1], 6)

                results.append(row)

                top3 = ", ".join(f"{w}({p:.4f})" for w, p, _ in predictions[:3])
                line = (
                    f"  [{i}/{len(words) - 1}] {actual_word} -> "
                    f"surp={out['surprisal_bits']:.1f}b "
                    f"cos={out['cos_dist_centroid']:.3f} "
                    f"rank={actual_rank if actual_rank > 0 else 'X'} "
                    f"[{n_preds} words, {prob_coverage:.0%}] | "
                    f"{top3}  ({elapsed:.1f}s)"
                )
                print(line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"), flush=True)
        except KeyboardInterrupt:
            print(f"\n  --- Interrupted after {len(results)} words ---")

        if results:
            ppl = 2 ** (sum(r["surprisal_bits"] for r in results) / len(results))
            avg_cos = sum(r["cos_dist_centroid"] for r in results) / len(results)
            print(f"\n  Perplexity (word-level): {ppl:.1f}")
            print(f"  Avg cos_dist_centroid: {avg_cos:.4f}")

        print(
            f"  Total: {total_time:.1f}s "
            f"({total_time / max(len(results), 1):.1f}s/word)"
        )
        self._print_profiling(f"{len(results)} words")
        return results


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    load_dotenv()
    os.makedirs(OUT_DIR, exist_ok=True)

    data_dir = os.path.abspath(DATA_DIR)
    txt_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {data_dir}")

    fname = txt_files[POEM_INDEX]
    poem_name = os.path.splitext(fname)[0]
    path = os.path.join(data_dir, fname)

    if RUN_MODE == "local":
        print(f"\nProcessing poem [{POEM_INDEX}]: {fname}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        rows = SemanticSurpriseAnalyser().analyze(text)
        df = pd.DataFrame(rows)

        out_path = os.path.join(OUT_DIR, f"{poem_name}_semantic.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path}")

    else:
        task = Task.init(
            project_name="PHD_Poetry",
            task_name="PPL_Semantic_Beam_SinglePoem",
        )
        task.add_requirements("spacy", ">=3.7.0,<3.8.0")
        task.set_comment(
            f"Single-poem semantic surprise + surprisal tables only. "
            f"nucleus_threshold={PROB_MASS_THRESHOLD}, beam={BEAM_WIDTH}, model={MODEL_NAME}"
        )

        if RUN_MODE == "remote":
            task.execute_remotely(queue_name="default", exit_process=True)

        logger = task.get_logger()

        print(f"\nProcessing poem [{POEM_INDEX}]: {fname}", flush=True)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        rows = SemanticSurpriseAnalyser().analyze(text)
        df = pd.DataFrame(rows)

        out_path = os.path.join(OUT_DIR, f"{poem_name}_semantic.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path}", flush=True)

        ppl = 2 ** (df["surprisal_bits"].mean())
        logger.report_scalar("poem_stats", "perplexity_word_level", ppl, 0)
        logger.report_scalar(
            "poem_stats", "avg_cos_dist_centroid", df["cos_dist_centroid"].mean(), 0
        )
        logger.report_scalar(
            "poem_stats", "pct_in_nucleus", df["in_nucleus"].mean() * 100, 0
        )
        task.upload_artifact("semantic_csv", out_path)

        # --- Report table to ClearML (full, including predictions) ---
        logger.report_table(
            "Results",
            "word_analysis",
            table_plot=df,
        )

        # --- Plotly plots to ClearML ---
        import plotly.graph_objects as go

        words = df["actual_word"].tolist()
        positions = df["word_position"].tolist()

        # Build hover text with top predictions for each word
        def _build_hover(row):
            preds = []
            for k in range(1, MAX_PREDICTIONS + 1):
                col_w, col_p = f"pred_{k}", f"prob_{k}"
                if col_w in row and pd.notna(row.get(col_w)):
                    preds.append(f"  {k}. {row[col_w]} ({row[col_p]:.4f})")
            pred_block = "<br>".join(preds[:10])  # show top-10 in hover
            return (
                f"<b>{row['actual_word']}</b> (pos {row['word_position']})<br>"
                f"surprisal={row['surprisal_bits']:.2f} bits<br>"
                f"cos_dist={row['cos_dist_centroid']:.4f}<br>"
                f"rank={row['actual_rank']}<br>"
                f"<b>predictions:</b><br>{pred_block}"
            )

        hover_texts = [_build_hover(row) for _, row in df.iterrows()]

        # 1) Z-score normalized surprisal + cos_dist on same scale
        surp = df["surprisal_bits"]
        cos_d = df["cos_dist_centroid"]
        surp_z = (surp - surp.mean()) / surp.std() if surp.std() > 0 else surp * 0
        cos_z = (cos_d - cos_d.mean()) / cos_d.std() if cos_d.std() > 0 else cos_d * 0
        divergence = surp_z - cos_z

        fig_norm = go.Figure()
        fig_norm.add_trace(
            go.Scatter(
                x=positions,
                y=surp_z.tolist(),
                mode="lines+markers",
                name="surprisal (z)",
                text=words,
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )
        fig_norm.add_trace(
            go.Scatter(
                x=positions,
                y=cos_z.tolist(),
                mode="lines+markers",
                name="cos_dist (z)",
                text=words,
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )
        fig_norm.add_trace(
            go.Bar(
                x=positions,
                y=divergence.tolist(),
                name="divergence (surp_z - cos_z)",
                marker_color=[
                    "rgba(255,100,100,0.4)" if d > 0 else "rgba(100,100,255,0.4)"
                    for d in divergence
                ],
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )
        fig_norm.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_norm.update_layout(
            title=f"Surprisal vs Cosine Distance (z-score) — {poem_name}",
            xaxis_title="Word position",
            yaxis_title="Z-score",
            barmode="overlay",
        )
        logger.report_plotly("Plots", "surprisal_vs_cos_zscore", fig_norm)

        # 2) Rank plot (where was the actual word in predictions?)
        ranks = df["actual_rank"].tolist()
        colors = ["green" if r > 0 else "red" for r in ranks]
        fig_rank = go.Figure(
            go.Bar(
                x=positions,
                y=[r if r > 0 else MAX_PREDICTIONS + 1 for r in ranks],
                text=words,
                marker_color=colors,
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )
        fig_rank.update_layout(
            title=f"Actual Word Rank in Predictions — {poem_name}",
            xaxis_title="Word position",
            yaxis_title="Rank (lower = more predicted)",
            yaxis=dict(autorange="reversed"),
        )
        logger.report_plotly("Plots", "actual_word_rank", fig_rank)

        # 3) Surprisal vs Cosine scatter (quadrant view) with predictions hover
        fig_scatter = go.Figure(
            go.Scatter(
                x=df["surprisal_bits"],
                y=df["cos_dist_centroid"],
                mode="markers+text",
                text=words,
                textposition="top center",
                textfont=dict(size=9),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )
        med_surp = df["surprisal_bits"].median()
        med_cos = df["cos_dist_centroid"].median()
        fig_scatter.add_hline(y=med_cos, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=med_surp, line_dash="dash", line_color="gray")
        fig_scatter.update_layout(
            title=f"Surprisal vs Cosine Distance — {poem_name}",
            xaxis_title="Surprisal (bits)",
            yaxis_title="Cosine distance (mean)",
        )
        logger.report_plotly("Plots", "surprisal_vs_cos_scatter", fig_scatter)

        print("  Reported 3 plots + table to ClearML", flush=True)

    print("\nDone.", flush=True)
