"""Pi0.5 prompt tokenization, equivalent to OpenPI's but without SentencePiece per call.

Pi0.5 puts the robot state inside the language prompt: it discretizes each
normalized state dimension into one of 256 bins and writes the bin indices out
as text (`models/tokenizer.py:23-30` upstream)::

    "Task: fold the towel, State: 128 64 243 -1;\\nAction: "

That makes tokenization part of every inference, because the state changes every
call -- unlike Pi0, whose prompt is fixed at load time. Running SentencePiece on
the full string costs ~41 us of host latency ahead of the graph replay.

It does not have to. The prompt is `head(task) + sum_d " {bin_d}" + tail`, and
for this tokenizer the segmentation of the number block does not cross value
boundaries: every value is preceded by a space, which becomes its own `U+2581`
piece, and digits are individual pieces::

    <bos> Task : |_fold |_the |_towel , |_State :     head, fixed per episode
    _ 1 2 8 | _ 6 4 | _ 2 4 3 | _- 1                  the state values
    ; \\n Action : _                                   tail, fixed forever

So the per-inference work reduces to 32 lookups in a 257-entry table plus a
concatenation. `Pi05Tokenizer.encode` is that; `Pi05Tokenizer.reference` is the
upstream implementation verbatim, kept here so the two can be compared. They
must agree exactly -- `eval.correctness.pi05.tokenize_parity` is the gate, and
it also pins the closed form in `discretize`.

The table is also the input the device-side version would need: `value_tokens`
and `value_lengths` are exposed for that, since the same decomposition is what
makes an in-graph tokenizer possible at all.

Everything here is host-side numpy; the engine copies the result into its static
`prompt_token_ids` buffer.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: OpenPI's `max_token_len` for Pi0.5 (`pi0_config.py:38`). The model output is
#: invariant to a larger pad and truncates at a smaller one, but this target
#: matches upstream exactly rather than picking its own shape profile.
MAX_TOKEN_LEN = 200

NUM_BINS = 256
#: `np.digitize` returns 0 for `state < -1`, which becomes -1 after the offset,
#: so the value alphabet is 257 symbols wide and `"-1"` is a legal state string.
BIN_MIN = -1
BIN_MAX = NUM_BINS - 1
NUM_VALUES = BIN_MAX - BIN_MIN + 1

_TAIL_TEXT = ";\nAction: "

#: The upstream bin edges, used only by the reference path.
_BINS = np.linspace(-1.0, 1.0, NUM_BINS + 1)[:-1]


def discretize(state: np.ndarray) -> np.ndarray:
    """Bin a normalized state exactly as `np.digitize(state, _BINS) - 1` does.

    Branch-free so the same expression can be lowered into a kernel later.

    The edges are `bins[j] = -1 + j/128`, so `bins[j] <= x` iff `j <= 128*x + 128`
    and the result is `clamp(floor(128*x) + 128, -1, 255)`. Note that the
    superficially equivalent `floor(128*(x+1))` is **wrong**: for `|x|` below the
    ulp of 1.0 the sum rounds to exactly 1.0 and the result is off by one,
    whereas `128*x` is an exact power-of-two scaling for every float. Verified
    bit-exact against `np.digitize` over 400k samples including every bin edge,
    in float32 and float64.

    NaN follows upstream into the top bin, because `np.digitize` is a
    `searchsorted` and NaN sorts above every edge.
    """
    scaled = np.floor(np.asarray(state) * 128.0) + 128.0
    scaled = np.where(np.isnan(scaled), float(BIN_MAX), scaled)
    return np.clip(scaled, BIN_MIN, BIN_MAX).astype(np.int32)


def clean_prompt(prompt: str) -> str:
    """OpenPI's prompt normalization (`models/tokenizer.py:24`)."""
    return prompt.strip().replace("_", " ").replace("\n", " ")


def _load_processor(tokenizer_path: str | Path | None):
    """Open the PaliGemma SentencePiece model, or explain how to get one."""
    try:
        import sentencepiece
    except ImportError as error:
        raise RuntimeError(
            "sentencepiece is required for Pi0.5 prompt tokenization; "
            "install it with `pip install -e '.[pi05]'`"
        ) from error

    if tokenizer_path is None:
        tokenizer_path = os.environ.get("PALIGEMMA_TOKENIZER")
    if tokenizer_path is None:
        raise ValueError(
            "no tokenizer model: pass tokenizer_path or set PALIGEMMA_TOKENIZER. "
            "The model is gs://big_vision/paligemma_tokenizer.model, the same file "
            "OpenPI downloads."
        )
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    with tokenizer_path.open("rb") as handle:
        return sentencepiece.SentencePieceProcessor(model_proto=handle.read())


class Pi05Tokenizer:
    """Tokenize a Pi0.5 prompt, reusing the same buffers on every call.

    `set_task` encodes the head once per episode; `encode` then costs a table
    lookup per state dimension. Both `encode` and `reference` return
    `(tokens[max_token_len] int32, mask[max_token_len] bool)`.

    OpenPI returns int64 tokens; int32 is used here because the value is on its
    way into a device buffer and the vocabulary is 257152 wide. The token
    *values* are identical, which is what the parity gate checks.
    """

    def __init__(self, tokenizer_path: str | Path | None = None,
                 max_token_len: int = MAX_TOKEN_LEN):
        self._processor = _load_processor(tokenizer_path)
        self._max_len = max_token_len

        pieces = [self._processor.encode(f" {value}")
                  for value in range(BIN_MIN, BIN_MAX + 1)]
        #: Token ids per bin value, indexed by `bin - BIN_MIN`, zero-padded.
        self.value_lengths = np.array([len(p) for p in pieces], dtype=np.int32)
        self.value_tokens = np.zeros((NUM_VALUES, int(self.value_lengths.max())),
                                     dtype=np.int32)
        for index, piece in enumerate(pieces):
            self.value_tokens[index, :len(piece)] = piece

        # Ragged form, which is what the concatenation actually gathers from.
        self._flat = np.concatenate([np.asarray(p, dtype=np.int32) for p in pieces])
        self._starts = np.concatenate(
            [[0], np.cumsum(self.value_lengths)])[:-1].astype(np.int32)

        self._tail = np.asarray(self._processor.encode(_TAIL_TEXT), dtype=np.int32)
        self._head: np.ndarray | None = None
        self._task: str | None = None

        self._scratch = np.zeros(max_token_len, dtype=np.int32)
        self._tokens = np.zeros(max_token_len, dtype=np.int32)
        self._mask = np.zeros(max_token_len, dtype=bool)

    @property
    def task(self) -> str | None:
        """The cleaned task string currently installed, or None."""
        return self._task

    @property
    def max_token_len(self) -> int:
        return self._max_len

    def set_task(self, prompt: str) -> None:
        """Install the task string, encoding the invariant head once.

        Call this whenever the task changes. `encode` reuses the result, so a
        task change without a `set_task` silently produces the previous task's
        tokens.
        """
        cleaned = clean_prompt(prompt)
        self._task = cleaned
        self._head = np.asarray(
            self._processor.encode(f"Task: {cleaned}, State:", add_bos=True),
            dtype=np.int32)

    def encode(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the installed task plus `state`, by table lookup.

        The returned arrays are internal buffers, overwritten by the next call.
        Copy them if you need to keep them.
        """
        if self._head is None:
            raise RuntimeError("call set_task(prompt) before encode(state)")

        index = discretize(state) - BIN_MIN
        lengths = self.value_lengths[index]
        if lengths.size:
            ends = np.cumsum(lengths)
            total = int(ends[-1])
            # For output position i belonging to dimension d, the source index is
            # starts[d] + (i - ends[d] + lengths[d]); repeat the per-dimension part
            # and add the running index.
            gather = (np.repeat(self._starts[index] - (ends - lengths), lengths)
                      + np.arange(total, dtype=np.int32))
        else:
            total, gather = 0, np.zeros(0, dtype=np.int32)

        head_len, tail_len = len(self._head), len(self._tail)
        length = head_len + total + tail_len
        if length > len(self._scratch):
            self._scratch = np.zeros(length, dtype=np.int32)
        self._scratch[:head_len] = self._head
        self._scratch[head_len:head_len + total] = self._flat[gather]
        self._scratch[head_len + total:length] = self._tail

        return self._finish(self._scratch[:length])

    def reference(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """OpenPI's `PaligemmaTokenizer.tokenize` verbatim, for comparison.

        Independent of `set_task`: it takes the prompt directly, so the gate
        cannot pass by accident through shared state.
        """
        cleaned_text = clean_prompt(prompt)
        discretized_state = np.digitize(state, bins=_BINS) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        tokens = self._processor.encode(full_prompt, add_bos=True)
        return self._finish(np.asarray(tokens, dtype=np.int32))

    def _finish(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pad or truncate to `max_token_len`, matching upstream's behaviour."""
        length = len(tokens)
        if length > self._max_len:
            logger.warning(
                "Token length (%d) exceeds max length (%d), truncating.",
                length, self._max_len)
        keep = min(length, self._max_len)
        self._tokens[:keep] = tokens[:keep]
        self._tokens[keep:] = 0
        self._mask[:keep] = True
        self._mask[keep:] = False
        return self._tokens, self._mask
