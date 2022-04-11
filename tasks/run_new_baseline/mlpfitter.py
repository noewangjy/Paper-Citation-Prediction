import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torch_f
import logging
import tqdm


class BasicBlock(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, dropout: float = 0.):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims)
        self.bn = nn.BatchNorm1d(output_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(torch.relu(self.bn(self.linear(x))))


def fit_mlp_classifier(model: nn.Module,
                       X_train,
                       Y_train,
                       lr=1e-4,
                       epochs=2,
                       batch_size=64,
                       device=torch.device('cpu'),
                       logger: logging.Logger = None,
                       update_freq: int = 100):
    model = model.to(device)
    model.train()
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train)
    if not isinstance(Y_train, torch.Tensor):
        Y_train = torch.tensor(Y_train)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    # if len(Y_train.shape) == 1 or Y_train.shape[1] == 1:
    #     Y_train = torch_f.one_hot(Y_train, num_classes=2).squeeze(1)
    if len(Y_train.shape) == 1 or Y_train.shape[1] == 1:
        Y_train = Y_train.unsqueeze(1)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size  # Drop last batch

    for epoch_idx in range(epochs):
        with tqdm.tqdm(range(num_batches)) as pbar:
            train_avg_loss = 0
            batch_idx_shuffled = torch.randperm(num_batches)
            for batch_idx in batch_idx_shuffled:
                X_train_sample = X_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                Y_train_sample = Y_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]

                optimizer.zero_grad()
                pred = model(X_train_sample)
                loss = torch_f.binary_cross_entropy_with_logits(pred.to(torch.float32), Y_train_sample.to(torch.float32))
                loss.backward()
                optimizer.step()
                train_avg_loss += float(loss.detach().cpu().numpy())
                if batch_idx % update_freq == 0 and batch_idx != 0:
                    pbar.set_description(f'epoch={epoch_idx}, step={num_batches * epoch_idx + batch_idx}, loss={loss.detach().cpu().numpy()}')
                    pbar.update(update_freq)

        if logger is not None:
            logger.info(f"epoch_idx={epoch_idx}, loss={train_avg_loss / num_batches}")
        else:
            print(f"epoch_idx={epoch_idx}, loss={train_avg_loss / num_batches}")

    return model


@torch.no_grad()
def infer_mlp_classifier(model: nn.Module,
                         X_test,
                         batch_size=64,
                         device=torch.device('cpu'),
                         update_freq: int = 100):
    model = model.to(device)
    model.eval()

    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_test = X_test.to(device)

    num_samples = X_test.shape[0]
    num_batches = num_samples // batch_size + 1
    all_scores = torch.empty(size=(num_samples,))

    with tqdm.tqdm(range(num_batches)) as pbar:
        pbar.set_description(f"Inferring")
        for batch_idx in range(num_batches):
            X_test_sample = X_test[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            pred = model(X_test_sample)
            # pred_max = torch.softmax(pred, dim=1)
            # pred_score = pred_max[:, 1]
            all_scores[batch_idx * batch_size: (batch_idx + 1) * batch_size] = pred.squeeze(1)
            if batch_idx % update_freq == 0 and batch_idx != 0:
                pbar.update(update_freq)

    return all_scores


if __name__ == '__main__':
    model = MODEL = nn.Sequential(
        BasicBlock(64, 28),
        BasicBlock(28, 1),
        nn.LogSigmoid()
    )
    x = torch.randn(1000, 64)
    y = torch.randint(0, 2, size=(1000,))

    device = torch.device('cuda:2')
    model = fit_mlp_classifier(model, x, y, epochs=10, device=device)

    scores = infer_mlp_classifier(model, x)

    print('finished')
