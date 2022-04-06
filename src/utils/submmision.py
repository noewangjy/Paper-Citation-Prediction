import csv
import os
from typing import Union

import numpy as np
import torch
import tqdm


def generate_submission(output_dir: str, pred: Union[np.ndarray, torch.Tensor], tag: str = ""):
    assert len(pred.shape) == 1 or (len(pred.shape) == 2 and pred.shape[1] == 0), ValueError(f"Expect pred to be a vector")
    if not pred.shape[0] == 106692:
        print(f"Expect pred to have length 106692 but have {pred.shape[0]}")

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = zip(range(len(pred)), pred)
    with tqdm.tqdm(range(len(pred))) as pbar:
        with open(os.path.join(output_dir, tag + '_submission_.csv'), 'w') as f:
            f.write('id,predicted\n')
            for idx, row in enumerate(results):
                f.write(f"{row[0]},{row[1]}\n")
                if idx % 1e3 == 0: pbar.update(1e3)


if __name__ == '__main__':
    print("testing np.ndarray")
    generate_submission('/', np.clip(np.abs(np.random.randn(106692)), 0, 1))
    print("testing torch.Tensor")
    generate_submission('/', torch.clip(torch.abs(torch.randn(106692)), 0, 1))
