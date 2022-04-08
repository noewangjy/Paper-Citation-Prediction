#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train Passage Matching Biencoder
"""

import logging
import os
import time
import timeit
from typing import Tuple, List
import numpy as np

import torch
from torch import Tensor as T
from torch import nn
from torch.distributed import init_process_group
# from transformers.utils import logging
from tqdm import tqdm, trange

from .modeling import BertEncoder, BertTensorizer
from .biencoder import (
    BiEncoder,
    BiEncoderLoss,
    BiEncoderBatch,
)
from .options import (
    set_seed,
    setup_logger,
)

from src.utils.submmision import generate_submission

from .data_utils import Tensorizer, DEFAULT_SELECTOR

from .model_utils import (
    move_to_device,
    CheckpointState,
    get_model_obj,
    load_states_from_checkpoint,
)

import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()
setup_logger(logger)


def get_optimizer(
        model: nn.Module,
        learning_rate: float = 1e-5,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


# # cfg = args.encoder
# def get_tensorizer(cfg):
#     sequence_length = cfg.sequence_length


class BiEncoderTrainer(object):

    def __init__(self, args) -> None:
        self.args = args
        self.tb_writer = SummaryWriter(args.train.output_dir)
        self.loss_fct = BiEncoderLoss()
        self.model: BiEncoder = None
        self.tensorizer: Tensorizer = None
        self.optimizer: torch.optim.Optimizer = None

        # Setup CUDA, GPU for distributed training
        if self.args.train.local_rank == -1 or self.args.train.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.args.train.no_cuda else "cpu")
            self.args.train.n_gpu = 0 if self.args.train.no_cuda else torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.args.train.local_rank)
            device = torch.device("cuda", self.args.train.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.args.train.n_gpu = 1
        self.device = device

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(mesage)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO if self.args.train.local_rank in [-1, 0] else logging.WARN,
        )

        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.args.train.local_rank,
            self.device.type,
            self.args.train.n_gpu,
            bool(self.args.train.local_rank != -1),
            self.args.train.fp16,
        )

        if self.args.train.local_rank in [-1, 0]:
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
        else:
            # Make sure only the first process in distributed training will download model and vocab
            torch.distributed.barrier()
        set_seed(self.args.train)

    @property
    def arguments(self):
        return self.args

    def reset_biencoder(self) -> None:
        self.model = None
        self.tensorizer = None

    def init_biencoder(self) -> Tuple[int, int]:
        saved_state = None
        if os.path.isfile(self.args.train.model_name_or_path):
            saved_state = load_states_from_checkpoint(self.args.train.model_name_or_path)
            self.args.biencoder = saved_state.biencoder_args
            logger.info("Using saved bi-encoder config: ", self.args.biencoder)
        else:
            self.args.biencoder.model_name_or_path = self.args.train.model_name_or_path

        cfg = self.args.biencoder
        logger.info("***** Initializing bi-encoder *****")

        # self.args.model_type = self.args.model_type.lower()
        query_encoder = BertEncoder(args=cfg)
        passage_encoder = BertEncoder(args=cfg)
        fix_q_encoder = cfg.fix_q_encoder if hasattr(cfg, "fix_q_encoder") else False
        fix_p_encoder = cfg.fix_p_encoder if hasattr(cfg, "fix_p_encoder") else False
        self.model = BiEncoder(query_model=query_encoder, passage_model=passage_encoder, fix_q_encoder=fix_q_encoder,
                               fix_p_encoder=fix_p_encoder)
        self.tensorizer = BertTensorizer(args=cfg, max_seq_len=cfg.max_sequence_length, pad_to_max=cfg.pad_to_max)
        self.model.to(self.device)
        if cfg.special_tokens:
            self.tensorizer.add_special_tokens(cfg.special_tokens)

        if saved_state:
            self.load_saved_state(saved_state)
            epochs_trained = saved_state.epoch
            model_name_tokens = self.args.train.model_name_or_path.split("/")[-1].split(".")
            steps_trained = int(model_name_tokens[-1]) if len(model_name_tokens) == 3 else 0
        else:
            epochs_trained = 0
            steps_trained = 0
        return epochs_trained, steps_trained

    def save_checkpoint(self, scheduler, epoch: int, tag: str = None) -> str:
        cfg = self.args.train

        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)

        if tag:
            output_dir = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch) + "." + tag)
        else:
            output_dir = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch))
        model_to_save = get_model_obj(self.model)
        state = CheckpointState(
            model_to_save.get_state_dict(),
            self.optimizer.state_dict(),
            epoch,
            self.args.biencoder
        )
        torch.save(state._asdict(), output_dir)
        logger.info(f"Saved checkpoint at: {output_dir}")
        return output_dir

    def load_saved_state(self, saved_state: CheckpointState):
        logger.info("Loading checkpoint @ epoch=%d", saved_state.epoch)

        model_to_load = get_model_obj(self.model)
        logger.info("Loading saved model state ...")

        model_to_load.load_state(saved_state, strict=True)
        logger.info("Saved state loaded")
        if self.optimizer:
            logger.info("Using saved optimizer state")
            self.optimizer.load_state_dict(saved_state.optimizer_dict)

    def train(self, train_dataset: Dataset):
        cfg = self.args.train
        train_batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)

        if cfg.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        # if max_steps is specified, determine training epochs by max_steps
        if cfg.max_steps > 0:
            total_steps = cfg.max_steps
            cfg.num_train_epochs = cfg.max_steps // (len(train_dataloader) // cfg.gradient_accumulation_steps) + 1
        else:
            # otherwise, determine total steps by training epochs
            total_steps = len(train_dataloader) // cfg.gradient_accumulation_steps * cfg.num_train_epochs

        global_epochs, global_steps = self.init_biencoder()

        self.optimizer = get_optimizer(self.model, cfg.learning_rate, cfg.adam_eps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
        )

        # Multi-GPU training
        if cfg.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        biencoder: BiEncoder = get_model_obj(self.model)

        # Distributed training
        if cfg.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True
            )

        # Train
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", cfg.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * cfg.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if cfg.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", total_steps)

        if os.path.isfile(cfg.model_name_or_path):
            logger.info("  Continuing training from epoch: %d, step: %d", global_epochs, global_steps)
        else:
            logger.info("  Starting fine-tuning from scratch")

        self.model.zero_grad()
        self.model.train()

        set_seed(cfg)
        steps_to_ignore = global_steps

        # Start loop for training
        for epoch in range(cfg.num_train_epochs):
            if global_epochs > 0:
                global_epochs -= 1
                continue
            epoch_loss = 0
            epoch_correct_predictions = 0

            logger.info("  Training Epoch: {}".format(epoch))
            steps_trained_in_current_epoch = 0
            with tqdm(train_dataloader, desc="Step iteration") as pbar:
                if steps_to_ignore != 0:
                    pbar.update(steps_to_ignore)
                    steps_to_ignore = 0

                for step, batch in enumerate(train_dataloader):

                    # Generate bi-encoder sample batch
                    biencoder_batch: BiEncoderBatch = biencoder.create_biencoder_input(
                        batch=batch,
                        tensorizer=self.tensorizer,
                    )

                    # Get representation position in input sequence
                    selector = DEFAULT_SELECTOR
                    rep_positions = selector.get_positions(biencoder_batch.query_ids, self.tensorizer)

                    # Do bi-encoder forward pass:
                    input_batch = BiEncoderBatch(**move_to_device(biencoder_batch._asdict(), self.device))

                    query_attention_mask = self.tensorizer.get_attention_mask(input_batch.query_ids)
                    passage_attention_mask = self.tensorizer.get_attention_mask(input_batch.context_ids)

                    outputs = self.model(
                        input_batch.query_ids,
                        input_batch.query_segments,
                        query_attention_mask,
                        input_batch.context_ids,
                        input_batch.context_segments,
                        passage_attention_mask,
                        encoder_type=cfg.encoder_type,
                        representation_token_pos=rep_positions
                    )

                    # Calculate NLL loss
                    loss, correct_cnt = self.loss_fct.calculate(
                        scores=outputs,
                        labels=input_batch.labels,
                        loss_scale=cfg.loss_scale
                    )
                    correct_cnt = correct_cnt.sum().item()

                    if cfg.n_gpu > 1:
                        loss = loss.mean()
                    if cfg.gradient_accumulation_steps > 1:
                        loss = loss / cfg.gradient_accumulation_steps

                    epoch_correct_predictions += correct_cnt
                    epoch_loss += loss.item()
                    # rolling_train_loss += loss.item()

                    # Do bi-encoder backward propagation
                    loss.backward()
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        scheduler.step()
                        self.model.zero_grad()

                    global_steps += 1
                    steps_trained_in_current_epoch += 1
                    pbar.set_description("Epoch: {} Steps: {}, BCE loss: {}".format(epoch, global_steps,
                                                                                    epoch_loss / steps_trained_in_current_epoch))
                    pbar.update()

                    self.tb_writer.add_scalar('train_step_loss', epoch_loss / steps_trained_in_current_epoch,
                                              global_steps)
                    self.tb_writer.add_text('train_step_loss', 'train_step_loss')

                    if global_steps >= total_steps:
                        logger.info("Training finished by max_steps: {}".format(cfg.max_steps))
                        self.save_checkpoint(scheduler, epoch=epoch, tag=str(cfg.max_steps))
                        return

            epoch_loss = epoch_loss / steps_trained_in_current_epoch
            epoch_acc = epoch_correct_predictions / len(train_dataset)
            logger.info("Av Loss per epoch=%f", epoch_loss)
            logger.info("Epoch total correct predictions = %f", epoch_acc)
            self.tb_writer.add_scalar("train_epoch_loss", epoch_loss, epoch)
            self.tb_writer.add_text("train_epoch_loss", "train_epoch_loss")
            self.tb_writer.add_scalar("train_epoch_accuracy", epoch_acc, epoch)
            self.tb_writer.add_text("train_epoch_accuracy", "train_epoch_accuracy")
            logger.info("Epoch finished on %d", cfg.local_rank)
            self.save_checkpoint(scheduler, epoch=epoch)
            self.tb_writer.flush()

    def evaluate(self, dev_dataset: Dataset, tag: str = None):
        logger.info("NLL validation on dev data ...")
        cfg = self.args.train
        if not self.model or not self.tensorizer:
            self.init_biencoder()

        # Multi-GPU training
        if cfg.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training
        if cfg.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True
            )

        self.model.eval()

        eval_batch_size = cfg.per_gpu_eval_batch_size * max(1, cfg.n_gpu)

        eval_sampler = SequentialSampler(dev_dataset)
        eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if cfg.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        biencoder: BiEncoder = get_model_obj(self.model)

        logger.info("***** Running evaluation {} on dev-data *****".format(tag))
        logger.info("  Num examples = %d", len(dev_dataset))
        logger.info("  Batch size = %d", eval_batch_size)

        total_loss = 0.0
        total_correct_predictions = 0
        steps_evaled = 0

        with tqdm(eval_dataloader, desc="Evaluating") as pbar:
            for step, batch in enumerate(eval_dataloader):
                # Generate bi-encoder sample batch
                biencoder_batch: BiEncoderBatch = biencoder.create_biencoder_input(
                    batch=batch,
                    tensorizer=self.tensorizer,
                )

                # Get representation position in input sequence
                selector = DEFAULT_SELECTOR
                rep_positions = selector.get_positions(biencoder_batch.query_ids, self.tensorizer)

                # Do bi-encoder forward pass:
                input_batch = BiEncoderBatch(**move_to_device(biencoder_batch._asdict(), self.device))

                query_attention_mask = self.tensorizer.get_attention_mask(input_batch.query_ids)
                passage_attention_mask = self.tensorizer.get_attention_mask(input_batch.context_ids)

                with torch.no_grad():
                    outputs = self.model(
                        input_batch.query_ids,
                        input_batch.query_segments,
                        query_attention_mask,
                        input_batch.context_ids,
                        input_batch.context_segments,
                        passage_attention_mask,
                        encoder_type=cfg.encoder_type,
                        representation_token_pos=rep_positions
                    )
                # Calculate NLL loss
                loss, correct_cnt = self.loss_fct.calculate(
                    scores=outputs,
                    labels=input_batch.labels,
                    loss_scale=cfg.loss_scale
                )
                correct_cnt = correct_cnt.sum().item()

                if cfg.n_gpu > 1:
                    loss = loss.mean()
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps

                total_loss += loss.item()
                total_correct_predictions += correct_cnt
                steps_evaled += 1
                pbar.set_description("Current loss: {}".format(total_loss / steps_evaled))
                pbar.update()

        total_loss = total_loss / len(dev_dataset)
        correct_ratio = float(total_correct_predictions / len(dev_dataset))
        logger.info(
            "BCE Validation: loss = %f. correct prediction ratio  %d/%d ~  %f",
            total_loss,
            total_correct_predictions,
            len(dev_dataset),
            correct_ratio,
        )
        with open(os.path.join(cfg.output_dir, "dev_result.txt"), "a") as f:
            f.write(f"{tag}: NLL loss: {total_loss}, accuracy: {correct_ratio}\n")

    def predict(self, test_dataset: Dataset, write_file: bool = False, tag: str = None) -> List[float]:
        cfg = self.args.train

        if not self.model or not self.tensorizer:
            self.init_biencoder()
        self.model.eval()

        eval_batch_size = cfg.per_gpu_eval_batch_size * max(1, cfg.n_gpu)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size)

        if cfg.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)
        biencoder: BiEncoder = get_model_obj(self.model)

        logger.info("***** Running Prediction {} on test-data *****".format(tag))
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", eval_batch_size)

        all_predictions = []

        for batch in tqdm(test_dataloader, desc="Predicting"):
            # Generate bi-encoder sample batch
            biencoder_batch: BiEncoderBatch = biencoder.create_biencoder_input(
                batch=batch,
                tensorizer=self.tensorizer,
            )

            # Get representation position in input sequence
            selector = DEFAULT_SELECTOR
            rep_positions = selector.get_positions(biencoder_batch.query_ids, self.tensorizer)

            # Do bi-encoder forward pass:
            input_batch = BiEncoderBatch(**move_to_device(biencoder_batch._asdict(), self.device))

            query_attention_mask = self.tensorizer.get_attention_mask(input_batch.query_ids)
            passage_attention_mask = self.tensorizer.get_attention_mask(input_batch.context_ids)

            with torch.no_grad():
                outputs = self.model(
                    input_batch.query_ids,
                    input_batch.query_segments,
                    query_attention_mask,
                    input_batch.context_ids,
                    input_batch.context_segments,
                    passage_attention_mask,
                    encoder_type=cfg.encoder_type,
                    representation_token_pos=rep_positions
                )

            # scores,size = (batch_size, batch_size)
            softmax_scores = nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(softmax_scores, dim=1)
            predictions = predictions.detach().cpu().numpy()
            all_predictions.extend(predictions)

        if write_file:
            generate_submission(cfg.output_dir, np.array(all_predictions), tag=tag)

        return all_predictions
