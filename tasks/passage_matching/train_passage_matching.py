import hydra
import torch
import os
from hydra.utils import to_absolute_path
import numpy as np

from torch.utils.data import Subset
from src.models.passage_matching.trainer import BiEncoderTrainer
from src.utils import NetworkDatasetPassageMatching


@hydra.main(config_path="conf", config_name="config")
def main(args):
    if not args.train.no_cuda:
        torch.backends.cudnn.benchmark = True

    args.data_path = to_absolute_path(args.data_path)
    args.train.output_dir = to_absolute_path(args.train.output_dir)

    trainer = BiEncoderTrainer(args=args)

    if args.do_train:
        train_data_path = os.path.join(args.data_path, args.train_file)
        train_dataset = NetworkDatasetPassageMatching(train_data_path)
        if args.train_dataset_size:
            train_dataset = Subset(train_dataset, np.arange(args.train_dataset_size)-int(args.train_dataset_size/2))
        trainer.train(train_dataset)

    if args.do_eval:
        dev_data_path = os.path.join(args.data_path, args.dev_file)
        dev_dataset = NetworkDatasetPassageMatching(dev_data_path)
        if args.dev_dataset_size:
            dev_dataset = Subset(dev_dataset, np.arange(args.dev_dataset_size) - int(args.dev_dataset_size/2))
        if args.eval_all_checkpoints:
            if not os.path.isdir(args.train.output_dir):
                raise ValueError("cfg.train.output_dir NOT exists!")
            elif len(os.listdir(args.train.output_dir)) == 0:
                raise ValueError("cfg.train.output_dir is EMPTY!")
            # Eval all checkpoints in train.output_dir
            for checkpoint in os.listdir(args.train.output_dir):
                checkpoint_tokens = checkpoint.split(".")
                if checkpoint_tokens[0] != args.train.checkpoint_file_name+"_"+args.machine.biencoder:
                    continue
                # Reset the model to load from checkpoint
                trainer.reset_biencoder()
                trainer.args.train.model_name_or_path = os.path.join(args.train.output_dir, checkpoint)
                trainer.evaluate(dev_dataset, tag=checkpoint)
        else:
            # Eval the last trained model
            trainer.init_biencoder()
            trainer.evaluate(dev_dataset, tag=args.train.model_name_or_path)

    if args.do_predict:
        test_data_path = os.path.join(args.data_path, args.test_file)
        test_dataset = NetworkDatasetPassageMatching(test_data_path)
        if args.predict_all_checkpoints:
            if not os.path.isdir(args.train.output_dir):
                raise ValueError("cfg.train.output_dir NOT exists!")
            elif len(os.listdir(args.train.output_dir)) == 0:
                raise ValueError("cfg.train.output_dir is EMPTY!")
            # Predict all checkpoints in cfg.train.output_dir
            for checkpoint in os.listdir(args.train.output_dir):
                # Reset the model to load from checkpoint
                trainer.reset_biencoder()
                trainer.args.train.model_name_or_path = os.path.join(args.train.output_dir, checkpoint)
                trainer.predict(test_dataset, tag=checkpoint)
        else:
            # Predict the last trained model
            trainer.init_biencoder()
            trainer.predict(test_dataset, tag=args.train.model_name_or_path)


if __name__ == "__main__":
    main()
