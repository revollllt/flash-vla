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
  per inference inside one captured graph. They matter where a kernel selects
  a row bucket from them.

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

On `rtx5090/pi05`, whose backbone buckets a 968-row prefix at M896 and M968
(`bucketed_backbone.py`), every RoboDojo observation lands in the M896 bucket.

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

```python
import sentencepiece
tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)  # PALIGEMMA_TOKENIZER
tokens = len(tokenizer.encode(instruction, add_bos=True) + tokenizer.encode("\n"))
```

## GR00T N1.7

| | `robodojo` (primary) | `libero` |
|---|---|---|
| Source | XPolicyLab `GR00T_N17`, `configs/robodojo_arx_x5_config.py` | GR00T-N1.7-LIBERO `libero_10` |
| Cameras | 3: front, left and right wrist, 256x256 | 2 |
| Vision patches / visual tokens | 768 / 192 | 512 / 128 |
| Text sequence (construction-time) | 206-237; declared 237, the longest instruction | 141-157 over the 40 tasks; declared 156, the prepared fixture's |
| Action horizon | 40 model rows. The data's 16-step horizon is padded to the model's `action_horizon=40` (`Gr00tN1d7.get_action` samples `[B, 40, action_dim]`) | 40 |

No Target builds `robodojo`: only the LIBERO observation is prepared as a
fixture. `robodojo` is declared for its graph and floor.

Sequence lengths come from the Cosmos-Reason2-2B tokenizer. Each is the visual
tokens (views x 64), plus the chat template's 9 tokens, plus the instruction's
tokens. The template's 9 tokens are the prepared LIBERO fixture's 28 non-visual
tokens less its instruction's 19:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(cosmos_reason2_2b, local_files_only=True)
sequence = views * 64 + 9 + len(tokenizer(instruction, add_special_tokens=False)["input_ids"])
```

## LingBot-VLA

| | `robodojo` (primary) |
|---|---|
| Source | XPolicyLab `LingBot_VLA`, `train_multinode_robodojo.sh`: `TOKENIZER_MAX_LEN=72`, `ACTION_DIM=14` in a 75-wide action space, chunk 50 by default |
| Cameras | 3 at 224x224, 64 visual tokens each |
| Prefix | 264 = 192 visual + 72 language slots |
| Action chunk / suffix rows | 50 / 51 |

These are the shapes `models/lingbot/spec.py` already fixes, so the workload
names no option. The recorded fixture (`synthetic=False`) and the seeded one
share them.

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
