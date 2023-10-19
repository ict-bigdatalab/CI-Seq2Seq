#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    BartConfig,
    BartTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
from tensorboardX import SummaryWriter
from bart_vae import BartForConditionalGeneration_VAE
from data_collator import DataCollatorForSeq2Seq_VAE
import json
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import jieba


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--kl_cost_annealing",
        action="store_true",
        help="Whether to apply KL cost annealing.",
    )
    parser.add_argument(
        "--training_vae",
        action="store_true",
        help="Whether to reparameterize during training, which allows for gradient flow, and whether to calculate kl_loss.",
    )
    parser.add_argument(
        "--fuse_seq_info",
        action="store_true",
        help="Whether to fuse a sequence information into one vector.",
    )
    parser.add_argument(
        "--start_training_vae_steps", type=int, default=0, help="Number of steps for setting training_vae False, which means learning mu only."
    )
    parser.add_argument(
        "--causal_latent_size",
        type=int,
        default=768,
        help="The embedding size of the causal latent variable.",
    )
    parser.add_argument(
        "--non_causal_latent_size",
        type=int,
        default=768,
        help="The embedding size of the non-causal latent variable.",
    )
    parser.add_argument(
        "--style_size",
        type=int,
        default=256,
        help="The embedding size of the style latent variable.",
    )
    parser.add_argument(
        "--u_size",
        type=int,
        default=32,
        help="The embedding size of the auxiliary variable.",
    )
    parser.add_argument(
        "--num_topics",
        type=int,
        default=10,
        help="The number of topic set for lda model.",
    )
    parser.add_argument(
        "--tc_th",
        type=float,
        default=0.02,
        help="Threshold value to choose causal topic.",
    )
    parser.add_argument(
        "--lda_dict_path", type=str, default=None, help="A dict model for LDA dictionary."
    )
    parser.add_argument(
        "--lda_model_path", type=str, default=None, help="A lda model for LdaModel."
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=10,
        help="The times to sample latent variables when initializing for evaluation.",
    )
    parser.add_argument(
        "--test_ep",
        type=int,
        default=50,
        help="The number of epochs to training latent variables for evaluation.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5,
        help="The number of interval epochs to save the training states.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="average pool",
        help="The method to aggregate sequence information.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--pretrained_model_dict_path",
        type=str,
        help="Path to pretrained model dict.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_for_latent",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for latent.",
    )
    parser.add_argument(
        "--learning_rate_ratio",
        type=float,
        default=10.0,
        help="Learning rate ratio for new parameters without pre-training.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--weight_decay_for_latent", type=float, default=0.005, help="Weight decay to use for latent.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--model_save_dir", type=str, default=None, help="Where to store the intermediate model.")
    parser.add_argument("--log_dir", type=str, default=None, help="Where to store the log file.")
    parser.add_argument("--predict_file", type=str, default=None, help="Where to store the predicted text.")
    parser.add_argument("--metric_file", type=str, default=None, help="Where to store the metric result.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument('--do_train', action="store_true", help="Whether or not to do training.")
    parser.add_argument(
        "--save_steps", type=int, default=0, help="Number of steps for saving a model."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Number of steps for writing to tensorboard."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    torch.set_num_threads(2)
    args = parse_args()
    dictionary = Dictionary.load(args.lda_dict_path)
    lda = LdaModel.load(args.lda_model_path)

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None and args.do_train:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = BartConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = BartConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.aggregate = args.aggregate
    config.training_vae = args.training_vae
    config.fuse_seq_info = args.fuse_seq_info
    config.causal_latent_size = args.causal_latent_size
    config.non_causal_latent_size = args.non_causal_latent_size
    config.style_size = args.style_size
    config.u_size = args.u_size
    config.max_source_length = args.max_source_length
    config.num_topics = args.num_topics
    config.tc_th = args.tc_th

    if args.tokenizer_name:
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.source_prefix is not None:
        special_tokens_dict = {'additional_special_tokens': ["<cls>"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        if not args.do_train:
            config.vocab_size = len(tokenizer)

    if args.model_name_or_path:
        model = BartForConditionalGeneration_VAE.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = BartForConditionalGeneration_VAE.from_config(config)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    logger.info(f"aggregate:{config.aggregate}")
    logger.info(f"training_vae:{config.training_vae}")
    logger.info(f"fuse_seq_info:{config.fuse_seq_info}")
    logger.info(f"causal_latent_size:{config.causal_latent_size}")
    logger.info(f"non_causal_latent_size:{config.non_causal_latent_size}")
    logger.info(f"style_size:{config.style_size}")
    logger.info(f"u_size:{config.u_size}")
    logger.info(f"num_topics:{config.num_topics}")
    logger.info(f"tc_th:{config.tc_th}")

    if args.do_train:
        # load weights to initialize model
        logger.info("Loading weights to initialize model...")
        pretrained_model_dict = torch.load(args.pretrained_model_dict_path + '/bart.pkl')
        new_state_dict = model.state_dict()
        for k, v in new_state_dict.items():
            if k not in pretrained_model_dict:
                if "_for_x" in k:
                    name = k.replace("_for_x","")
                    new_state_dict[k] = pretrained_model_dict[name]
        # load params
        model.load_state_dict(new_state_dict)

        logger.info("Checking parameters of decoder_for_x...")
        for t1,t2 in zip(model.model.decoder.named_parameters(),model.model.decoder_for_x.named_parameters()):
            t1_name, t1_param = t1
            t2_name, t2_param = t2
            assert t1_param.data.equal(t2_param.data)

        logger.info("Checking parameters of lm_head_for_x...")
        for t1,t2 in zip(model.lm_head.named_parameters(),model.lm_head_for_x.named_parameters()):
            t1_name, t1_param = t1
            t2_name, t2_param = t2
            assert t1_param.data.equal(t2_param.data)

        print("new parameters:")
        for k, v in new_state_dict.items():
            if k not in pretrained_model_dict:
                if "_for_x" not in k:
                    print(k)

    model.resize_token_embeddings(len(tokenizer))
    if args.do_train and args.source_prefix is not None:
        # expand lm_head_for_x
        extra_weight = model.lm_head.weight[-1].unsqueeze(0)
        new_weight = torch.cat([model.lm_head_for_x.weight, extra_weight], dim=0)
        model.lm_head_for_x.weight=torch.nn.Parameter(new_weight)

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["validation"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.save_steps > 0:
        if args.model_save_dir is None:
            raise ValueError(
                "Need an `model_save_dir` to save models when `save_steps` > 0."
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function_vae_lda_cr(examples):
        inputs = examples[text_column]

        docs_bow_list = [dictionary.doc2bow(jieba.lcut(doc)) for doc in inputs]
        topics_docs = lda.get_document_topics(docs_bow_list, minimum_probability=0)
        topic_distribution = [[topic[1] for topic in topics] for topics in topics_docs]

        targets = examples[summary_column]
        # cr = np.array([0.7]*len(inputs))
        cr = np.array([max(1,len(summary)) for summary in targets]) / np.array([max(1,len(doc)) for doc in inputs])
        # cr = np.array([max(1,len(summary.split())) for summary in targets]) / np.array([max(1,len(doc.split())) for doc in inputs])
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        # Setup the tokenizer for targets_for_x
        with tokenizer.as_target_tokenizer():
            labels_for_x = tokenizer(examples[text_column], max_length=args.max_source_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels_for_x["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_for_x["input_ids"]
            ]

        model_inputs["labels_for_x"] = labels_for_x["input_ids"]
        model_inputs["topic_distribution"] = topic_distribution
        model_inputs["cr"] = cr

        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function_vae_lda_cr,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache, # 
            desc="Running tokenizer on dataset",
        )

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq_VAE(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    if args.do_train:
        train_dataset = processed_datasets["train"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
    
    eval_dataset = processed_datasets["validation"]
    # Log a few random samples from the validation set:
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    if args.do_train:
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]

        new_params_no_decay = []
        new_params_decay = []
        backbone_params_no_decay = []
        backbone_params_decay = []
        for n, p in model.named_parameters():
            if (n not in pretrained_model_dict and '_for_x' not in n):
                if any(nd in n for nd in no_decay):
                    new_params_no_decay += [p]
                else:
                    new_params_decay += [p]
            else:
                if any(nd in n for nd in no_decay):
                    backbone_params_no_decay += [p]
                else:
                    backbone_params_decay += [p]

        optimizer_grouped_parameters = [
            {
                "params": new_params_decay,
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate_ratio * args.learning_rate,
            },
            {
                "params": new_params_no_decay,
                "weight_decay": 0.0,
                "lr": args.learning_rate_ratio * args.learning_rate,
            },
            {
                "params": backbone_params_decay,
                "weight_decay": args.weight_decay,
            },
            {
                "params": backbone_params_no_decay,
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    else:
        model, eval_dataloader = accelerator.prepare(
            model, eval_dataloader
        )        

    # Metric
    metric = load_metric("rouge")

    if args.do_train:

        # Train!
        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        train(args, model, train_dataloader, eval_dataloader, accelerator, tokenizer, metric, optimizer)
    else:
        evaluate(args, model, eval_dataloader, accelerator, tokenizer, metric)
    
def train(args, model, train_dataloader, eval_dataloader, accelerator, tokenizer, metric, optimizer):

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    tb_writer = SummaryWriter(args.log_dir)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    nb_tr_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        model.model.training_vae = True if args.training_vae and completed_steps >= args.start_training_vae_steps else False
        logger.info(f"training_vae:{model.model.training_vae}")
        for step, batch in enumerate(train_dataloader):
            outputs = model(target = "both", **batch)
            # loss = outputs.loss
            prediction_loss = outputs.prediction_loss
            reconstruction_loss = outputs.reconstruction_loss
            kl_loss = outputs.kl_loss
            topic_constraint = outputs.topic_constraint
            if kl_loss is not None:
                kl_coeff = (completed_steps / args.max_train_steps) if args.kl_cost_annealing else 1 # KL cost annealing
                loss = prediction_loss + reconstruction_loss + kl_coeff * kl_loss
            else:
                loss = prediction_loss + reconstruction_loss
            
            if topic_constraint is not None:
                loss = loss + topic_constraint

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            tr_loss += loss.item()
            nb_tr_steps += 1

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if args.save_steps > 0 and completed_steps % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(model, args.model_save_dir, completed_steps)

                if args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                    mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                    tb_writer.add_scalar('train/mean_loss', round(mean_loss,4), completed_steps)
                    cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('train/cur_loss', cur_loss, completed_steps)
                    logging_loss = tr_loss

                model.model.training_vae = True if args.training_vae and completed_steps >= args.start_training_vae_steps else False

            if completed_steps >= args.max_train_steps:
                break

        save_model(model, args.model_save_dir, completed_steps)
        
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch}
            path_checkpoint = args.model_save_dir + "/checkpoint_{}_epoch.pkl".format(epoch)
            torch.save(checkpoint, path_checkpoint)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def evaluate(args, model, eval_dataloader, accelerator, tokenizer, metric):
    model.eval()

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else args.max_length,
        "num_beams": args.num_beams,
    }

    def get_loss_batch(lm_logits_for_x, labels_for_x):
        loss_fct_vocab = torch.nn.CrossEntropyLoss(reduction="none")
        loss_vocab = loss_fct_vocab(lm_logits_for_x.view(-1, model.config.vocab_size), labels_for_x.view(-1)).view(lm_logits_for_x.size(0), -1)
        loss_batch = loss_vocab.sum(1) / torch.count_nonzero(loss_vocab, dim = 1)
        return loss_batch

    for step, batch in enumerate(tqdm(eval_dataloader, desc="eval")):
        # training for reconstruction
        # initialize latent factor
        with torch.no_grad():
            latent_init = None
            for ss in range(args.sample_num):
                outputs_for_x = model(
                    target="reconstruction",
                    cr=batch["cr"],
                    subset_id=batch["subset_id"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels_for_x=batch["labels_for_x"],
                    decoder_input_ids_for_x=batch["decoder_input_ids_for_x"],
                    )
                
                if latent_init is None:
                    latent_init = outputs_for_x.latent_result
                    min_rec_loss = get_loss_batch(outputs_for_x.reconstruction_logits, batch["labels_for_x"])
                else:
                    new_loss = get_loss_batch(outputs_for_x.reconstruction_logits, batch["labels_for_x"])
                    for i in range(batch["input_ids"].size(0)): # for each case in batch
                        if new_loss[i].item() < min_rec_loss[i].item():
                            min_rec_loss[i] = new_loss[i]
                            latent_init.sample_z[i] = outputs_for_x.latent_result.sample_z[i]
                            latent_init.z_mu[i] = outputs_for_x.latent_result.z_mu[i]
                            latent_init.z_logvar[i] = outputs_for_x.latent_result.z_logvar[i]

        # optimize
        init_latent_result = latent_init
        init_latent_result.sample_z.requires_grad = True
        optimizer_grouped_parameters_for_latent = [
            {
                "params": [init_latent_result.sample_z],
                "weight_decay": args.weight_decay_for_latent,
            }
        ]
        optimizer_for_latent = AdamW(optimizer_grouped_parameters_for_latent, lr=args.learning_rate_for_latent)
        model, optimizer_for_latent = accelerator.prepare(
            model, optimizer_for_latent
        )
        for i in range(args.test_ep):
            optimizer_for_latent.zero_grad()

            outputs_for_x_with_latent = model(
                target="reconstruction",
                cr=batch["cr"],
                subset_id=batch["subset_id"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels_for_x=batch["labels_for_x"],
                decoder_input_ids_for_x=batch["decoder_input_ids_for_x"],
                latent_result=init_latent_result,
                )

            rec_loss = outputs_for_x_with_latent.reconstruction_loss
            rec_loss.backward()
            optimizer_for_latent.step()

        # generation
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                target="prediction",
                cr=batch["cr"],
                subset_id=batch["subset_id"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                latent_result=init_latent_result,
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            if args.predict_file is not None:
                # save predict result
                for pred, label in zip(decoded_preds, decoded_labels):
                    predict_result = json.dumps({"pred":pred, "label":label})
                    with open(args.predict_file, 'a+') as fout:
                        fout.write(predict_result + '\n')

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    result = metric.compute(use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    logger.info(result)
    
    if args.metric_file is not None:
        # save metric result
        with open(args.metric_file, 'a+') as fout:
            fout.write(json.dumps(result))
    
    return(result)



def save_model(model, save_dir, global_step):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = save_dir + "/pytorch_model_{}.bin".format(global_step) 
    torch.save(model_to_save.state_dict(), str(output_model_file))

    
    
if __name__ == "__main__":
    main()