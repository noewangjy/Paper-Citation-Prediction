import hydra
import torch
import os

from src.models.passage_matching.trainer import BiEncoderTrainer
from src.utils import NetworkDatasetPassageMatching


@hydra.main(config_path="conf", config_name="config")
def main(args):
    if not args.train.no_cuda:
        torch.backends.cudnn.benchmark = True

    trainer = BiEncoderTrainer(args=args)

    if args.do_train:
        train_data_path = os.path.join(args.data_path, args.train_file)
        train_dataset = NetworkDatasetPassageMatching(train_data_path)
        trainer.train(train_dataset)

    if args.do_eval:
        dev_data_path = os.path.join(args.data_path, args.dev_file)
        dev_dataset = NetworkDatasetPassageMatching(dev_data_path)
        if args.eval_all_checkpoints:
            if not os.path.isdir(args.train.output_dir):
                raise ValueError("cfg.train.output_dir NOT exists!")
            elif len(os.listdir(args.train.output_dir)) == 0:
                raise ValueError("cfg.train.output_dir is EMPTY!")
            # Eval all checkpoints in train.output_dir
            for checkpoint in os.listdir(args.train.output_dir):
                # Reset the model to load from checkpoint
                trainer.reset_biencoder()
                trainer.args.train.model_name_or_path = os.path.join(args.train.output_dir, checkpoint)
                trainer.evaluate(dev_dataset, tag=checkpoint)
        else:
            # Eval the last trained model
            trainer.evaluate(dev_dataset, tag="last_trained_model")

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
            trainer.predict(test_dataset, tag="last_trained_model")



if __name__ == "__main__":
    main()
