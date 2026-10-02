# Workloads

A workload is one deployment of a model: the construction options that fix its
shape profile (cameras, action chunk, prompt capacity), and the fixture
parameters that shape depends on. Each model declares its workloads in
`src/flash_vla/models/<model>/definition.py` (`ModelDefinition.workloads`,
the primary first). Each Target names the workloads it builds and checks
(`Target.workloads`, its default first; `python -m tests.targets` declares
each one). A Target can still declare any workload of its model, for its graph
and its floor (`python -m measurement.work`). Harnesses select a workload with
`--workload <name>`. This page records where every value comes from; the code
declaration is authoritative for the values themselves.

Depth is not part of a workload. The denoising steps and layers are the
model's, the same in every benchmark here. A harness cuts them only to bisect,
and the identity records the depth it ran.

Benchmarks only fix shapes here. Nothing on this page needs a benchmark's
checkpoint, dataset or simulator: latency and floor do not depend on weight
values, and correctness is checked against the model's reference.

Two kinds of axis:

- **Construction-time** axes (cameras, chunk, prompt slots, a text sequence
  length) fix the graph. A different value is another construction, and
  another routing decision.
- **Replay-time** axes (how many prompt slots a given observation fills) vary
  per inference inside one construction. A model declares its one replay-time
  axis (`ModelDefinition.replay_axis`, `runtime/replay.py`) and each workload
  the axis's range over its observations (`Workload.replay_range`). A Target
  with a row granularity (`Target.replay_granularity`) captures the stages the
  axis reaches once per bucket of that many rows the range reaches, plus the
  full one, and each inference replays the bucket its length falls in.

Sources, read-only:

- RoboDojo: `robodojo-benchmark/RoboDojo@726e9aab`, with task instructions from
  `task/RoboDojo/tasks/*.py` `gen_instruction`. All 43 simulation tasks run on
  ARX X5. The instruction bounds are:
  - shortest: "Fold the clothes neatly." (fold_clothes);
  - longest: "Lift the basket more than 8 cm, identify the target object on
    the conveyor according to the image on the board, pick it up, and place it
    into the basket." (pick_from_conveyor_by_image).
- XPolicyLab (the RoboDojo policy adapters): `XPolicyLab/XPolicyLab@d6332bf1`,
  `policy/<model>/`.
- LIBERO: the 40 task strings of `libero_{spatial,object,goal,10}` BDDL files.

Prompt-token ranges come from each model's own tokenizer, on the CPU, with the
code under each table.

## Pi0.5

| | `robodojo` (primary) | `libero` |
|---|---|---|
| Source | XPolicyLab `Pi_05`, train config `pi05_base_aloha_full_sim_arx-x5_seed_0` = `Pi0Config(pi05=True)` defaults | OpenPI `pi05_libero` |
| Cameras | 3: `cam_high`, left and right wrist | 2 |
| Action chunk (expert rows M) | 50 | 10 |
| Prompt slots | 200 (`max_token_len`) | 200 |
| Prompt | `Task: <instruction>, State: <14 values>;\nAction: `. OpenPI tokenizes the 14-dim ALOHA state before padding it to 32 (`TokenizePrompt` precedes `PadStatesAndActions`), so `robot_state_dim=14` | task text only (`discrete_state=False`) |
| Prefix rows, physical | 968 = 3 x 256 + 200 | 712 = 2 x 256 + 200 |
| Valid prompt tokens (replay-time) | 43-101 | 6-22 |
| Valid prefix rows | 811-869 | 518-534: at least 178 of 712 rows (25%) are padding |

`robot_state_dim` and `discrete_state` shape the prompt, and so the valid rows.
The identity does not record them; the fixture's digest does.

The replay-time axis is `prompt_tokens`, the valid prompt tokens after the
image tokens; `replay_range` is the valid token range above. On
`rtx5090/pi05` (64-row granularity) the backbone is captured at 832, 896 and
968 prefix rows for `robodojo` (prompt buckets 64, 128, 200) and at 576 and 712
for `libero` (64, 200). The construction option
`prompt_tokens=N` replaces the tokenized prompt by N seeded tokens, to measure
or check one bucket.

The bounds combine the per-value token length (2 to 4, over bins -1..255) with
the shortest and longest instructions:

```python
import numpy as np
from flash_vla.models.pi05.tokenize import BIN_MIN, Pi05Tokenizer, TaskTokenizer

pi05 = Pi05Tokenizer(tokenizer_path)                 # PALIGEMMA_TOKENIZER
shortest_value = int(np.argmin(pi05.value_lengths)) + BIN_MIN
longest_value = int(np.argmax(pi05.value_lengths)) + BIN_MIN

def state_in_bin(value: int) -> np.ndarray:          # 14 normalized values in bin `value`
    return np.full(14, -1 + (value + 0.5) / 128 if value >= 0 else -1.5, dtype=np.float32)

for instruction in (shortest, longest):
    pi05.set_task(instruction)
    print([int(pi05.encode(state_in_bin(v))[1].sum()) for v in (shortest_value, longest_value)])

libero = TaskTokenizer(tokenizer_path)               # per LIBERO task string
libero.set_task(task); print(int(libero.mask.sum()))
```

## Pi0

| | `robodojo` (primary) |
|---|---|
| Source | XPolicyLab `Pi_0`, train config `pi0_base_aloha_full_sim_arx-x5_seed_0` = `Pi0Config()` defaults |
| Cameras | 3 |
| Action chunk | 50 (expert rows 51 with the state token) |
| Prompt | the instruction and a newline, 7-37 tokens. OpenPI pads to 48 and masks the padding; this engine bakes the prompt into `language_embeds` at load time and runs its valid rows only. The workload declares the longest instruction's 37 |
| Prefix rows | 805 = 3 x 256 + 37; the range 775-805 lies inside one 128-row tile (768-896) |
| Replay-time axis | none: the prompt is baked in at load time |

```python
import sentencepiece
tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)  # PALIGEMMA_TOKENIZER
tokens = len(tokenizer.encode(instruction, add_bos=True) + tokenizer.encode("\n"))
```

## GR00T N1.7

| | `robodojo` (primary) | `libero` |
|---|---|---|
| Source | XPolicyLab `GR00T_N17`, `configs/robodojo_arx_x5_config.py` | GR00T-N1.7-LIBERO `libero_10` |
| Cameras | 3: front, left and right wrist; the official 640x480 observations become 256x352 after aspect-preserving processing | 2 at 256x256 |
| Vision patches / visual tokens | 1056 / 264; patch grid `(1, 16, 22)` per camera | 512 / 128 |
| Text sequence (construction-time) | 280 for the prepared official demo frame 0 | 141-157 over the 40 tasks; declared 156, the prepared fixture's |
| Action horizon | 40 model rows. The data's 16-step horizon is padded to the model's `action_horizon=40` (`Gr00tN1d7.get_action` samples `[B, 40, action_dim]`) | 40 |
| Replay-time axis | `valid_sequence_tokens`, the valid tokens of `attention_mask`; no Target buckets it | same |

Thor builds `robodojo` with the official seed-0 checkpoint and a prepared
observation from `RoboDojo-Benchmark/RoboDojo@35efbc7d`,
`data/demo/arx_x5/data/episode_0000000.hdf5`, frame 0. NVIDIA's processor at
`Isaac-GR00T@51d4c89` uses the checkpoint's statistics and `new_embodiment`
projector 10. The XPolicyLab adapter at `bb9a0b5` passes RGB images without
square resizing; the processor preserves their aspect ratio.

Sequence lengths come from the Cosmos-Reason2-2B tokenizer. Each is the visual
tokens, plus the chat template's `5 + 2 * views` tokens, plus the instruction's
tokens. The LIBERO template's 9 tokens are the prepared fixture's 28 non-visual
tokens less its instruction's 19:

```python
from transformers import AutoTokenizer
import re
tokenizer = AutoTokenizer.from_pretrained(cosmos_reason2_2b, local_files_only=True)
# RoboDojo's checkpoint has formalize_language=True.
instruction = re.sub(r"[^\w\s]", "", instruction.lower())
sequence = 264 + 11 + len(tokenizer(instruction, add_special_tokens=False)["input_ids"])
```

The documented shortest/longest instructions use 4/31 instruction tokens,
giving 279/306 sequence tokens. The prepared demo has 5 instruction tokens,
so this fixed-shape target currently runs that 280-token observation. Other
instructions require their own sequence-length construction; padding changes
the unmasked vision-language refiner's result.

## LingBot-VLA

| | `robodojo` (primary) |
|---|---|
| Source | XPolicyLab `LingBot_VLA`, `train_multinode_robodojo.sh`: `TOKENIZER_MAX_LEN=72`, `ACTION_DIM=14` in a 75-wide action space, chunk 50 by default |
| Cameras | 3 at 224x224, 64 visual tokens each |
| Prefix | 264 = 192 visual + 72 language slots |
| Action chunk / suffix rows | 50 / 51 |
| Replay-time axis | `valid_language_tokens`, the valid slots of `language_masks`; no Target buckets it |

These are the shapes `models/lingbot/spec.py` already fixes, so the workload
names no option. The recorded fixture (`synthetic=False`) and the seeded one
share them.
The official pretrained and Robotwin post-trained LingBot-VLA checkpoints
load at this shape. RoboDojo's published `Lingbot_VA` checkpoint is a different
model; no RoboDojo-official LingBot-VLA fine-tune is available in that dataset.

## Adding a workload

1. Declare it in the model's `workloads`. Its options are `runner_source`
   keyword options, and they must fix every axis of the shape profile the
   benchmark sets.
2. Name it in `Target.workloads` of each Target that can construct it and has
   a fixture for it.
3. Add its column here: the source and revision, every axis, and the
   replay-time ranges with the code that computed them.
4. `tests/test_workloads.py` checks that every declared workload appears on
   this page.
