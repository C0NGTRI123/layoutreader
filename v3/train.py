import os
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
from loguru import logger
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer import Trainer
from huggingface_hub import snapshot_download

from helpers import DataCollator, MAX_LEN
from model.layoutlmv3 import LayoutLMv3ForTokenClassification


@dataclass
class Arguments(TrainingArguments):
    model_dir: str = field(
        default=None,
        metadata={"help": "Path to model, based on `microsoft/layoutlmv3-base`"},
    )
    dataset_dir: str = field(
        default=None,
        metadata={"help": "Path to dataset"},
    )


def load_train_and_dev_dataset(path: str, sample_ratio: float=1.0) -> (Dataset, Dataset):
    datasets = load_dataset(
        "json",
        data_files={
            "train": os.path.join(path, "train.jsonl.gz"),
            "dev": os.path.join(path, "dev.jsonl.gz"),
        },
    )

    if sample_ratio == 1.0:
        return datasets["train"], datasets["dev"]
    
    train_size = int(len(datasets["train"]) * sample_ratio)
    dev_size = int(len(datasets["dev"]) * sample_ratio)
    
    # Shuffle and select subset
    sampled_train = datasets["train"].select(range(train_size))
    sampled_dev = datasets["dev"].select(range(dev_size))
    
    return sampled_train, sampled_dev

def download_last_checkpoint(repo_id: str, token: str, local_dir: str):
    # Download specific checkpoint
    checkpoint_path = f"{local_dir}/last-checkpoint"
    if not os.path.exists(checkpoint_path):
        repo_path = snapshot_download(
            repo_id=repo_id,
            revision="main",
            token=token,
            allow_patterns=[f"last-checkpoint/*"],
            local_dir=local_dir
        )
    return checkpoint_path


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    # config add resume training from huggingface hub
    if args.resume_from_checkpoint:
        # Resume training path
        checkpoint_path = "checkpoint/last-checkpoint"
        if not os.path.exists(checkpoint_path):
            last_checkpoint = download_last_checkpoint(
                repo_id=args.model_dir,
                token="",
                local_dir="checkpoint"
            )
        model_path = last_checkpoint
    else:
        # Fresh training path
        model_path = args.model_dir

    train_dataset, dev_dataset = load_train_and_dev_dataset(args.dataset_dir, sample_ratio=1.0)
    logger.info(
        "Train dataset size: {}, Dev dataset size: {}".format(
            len(train_dataset), len(dev_dataset)
        )
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_dir, num_labels=MAX_LEN, visual_embed=False
    )
    # model.gradient_checkpointing_enable()
    data_collator = DataCollator()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
