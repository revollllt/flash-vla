"""Prepare real LIBERO observations with NVIDIA's evaluation-mode processor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd
import torch
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor


def prepare(checkpoint: Path, dataset: Path, output: Path, *, backbone: str, frame: int = 0) -> Path:
    """Save one episode-0 observation as tensors, plus its processor output and provenance."""
    processor = Gr00tN1d7Processor.from_pretrained(checkpoint, model_name=backbone)
    processor.eval()
    rows = pd.read_parquet(dataset / "data/chunk-000/episode_000000.parquet")
    row = rows.iloc[frame]
    metadata = json.loads((dataset / "meta/modality.json").read_text())
    tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()]
    task = next(item["task"] for item in tasks if item["task_index"] == int(row["task_index"]))
    state = {name: np.asarray(row["observation.state"])[spec["start"]:spec["end"]][None].astype(np.float32)
             for name, spec in metadata["state"].items()}
    images = {}
    for name, spec in metadata["video"].items():
        path = dataset / "videos/chunk-000" / spec["original_key"] / "episode_000000.mp4"
        with av.open(str(path)) as video:
            image = next((value for index, value in enumerate(video.decode(video=0))
                          if index == frame), None)
            if image is None:
                raise RuntimeError(f"Cannot decode frame {frame}: {path}")
            images[name] = [image.to_ndarray(format="rgb24")]
    step = VLAStepData(images=images, states=state, actions={}, text=task,
                       embodiment=EmbodimentTag.LIBERO_PANDA)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    collated = processor.collator([processed])["inputs"]
    inputs = {name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
              for name, tensor in collated.items() if isinstance(tensor, torch.Tensor)}
    config = Qwen3VLConfig.from_pretrained(backbone, local_files_only=True)
    inputs["position_ids"], _ = Qwen3VLModel.get_rope_index(
        SimpleNamespace(config=config), input_ids=inputs["input_ids"],
        image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])
    inputs["image_indices"] = torch.nonzero(inputs["input_ids"][0] == config.image_token_id).flatten()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"episode-0-frame-{frame}.pt"
    torch.save({"inputs": inputs, "states": {name: torch.from_numpy(value) for name, value in state.items()},
                "images": {name: torch.from_numpy(value[0]) for name, value in images.items()},
                "prompt": task, "episode": 0, "frame": frame,
                "source_revision": "51d4c89f72fda44cbf77285c6a8114b52676b8a1"}, path)
    print(json.dumps({"fixture": str(path), "prompt": task,
                      "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                 for name, value in inputs.items()},
                      "image_grid_thw": inputs["image_grid_thw"].tolist()}, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", required=True, help="Local Cosmos-Reason2-2B snapshot")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    prepare(args.checkpoint, args.dataset, args.output, backbone=args.backbone, frame=args.frame)
