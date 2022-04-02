import argparse

import torch
import torch.nn as nn

from src.utlis import evaluate_model_checkpoint, generate_submission
from src.models import PseudoModel


class PseudoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(2, 1)) / 2

    def forward(self, x):
        return x @ self.w


def main(args):
    model = PseudoModel()
    pseudo_data = torch.randn(106692, 2)
    pred = evaluate_model_checkpoint(model, args.checkpoint, pseudo_data)
    generate_submission('./', pred)


if __name__ == '__main__':
    pseudo_model = PseudoModel()
    torch.save(pseudo_model.state_dict(), './pseudo_checkpoint.pth')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--testset', type=str)
    main(parser.parse_args())
