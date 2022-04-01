from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def evaluate_model(model: nn.Module,
                   test_dataset: Dataset,
                   device=torch.device('cpu')) -> torch.Tensor:
    model.eval()
    model.to(device)
    result_list: List[torch.Tensor] = []

    for data in test_dataset:
        result_list.append(model(data))

    result: torch.Tensor = torch.cat(result_list, dim=0)
    result = torch.squeeze(result)

    return result


def evaluate_model_checkpoint(model: nn.Module,
                              checkpoint_path: str,
                              test_dataset: Dataset,
                              device=torch.device('cpu')) -> torch.Tensor:
    model.load_state_dict(torch.load(checkpoint_path))

    return evaluate_model(model, test_dataset, device)


def evaluate_model_pkl(pkl_path: str,
                       test_dataset: Dataset,
                       device=torch.device('cpu')) -> torch.Tensor:
    model = torch.load(pkl_path)

    return evaluate_model(model, test_dataset, device)
