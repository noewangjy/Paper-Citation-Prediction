import pickle
import numpy as np
import numba

from src.utils.io import check_md5
from src.utils.fitter import fit_lr_classifier

@numba.jit
def generate_array(cora_hidden_state, u_array, v_array):
    num_edges: int = u_array.shape[0]
    res: np.ndarray = np.empty(shape=(num_edges, cora_hidden_state.shape[1]))
    for idx in range(num_edges):
        u, v = u_array[idx], v_array[idx]
        h_u, h_v = cora_hidden_state[u], cora_hidden_state[v]
        res[idx] = h_u * h_v / (1e-8 + (np.linalg.norm(h_u) * np.linalg.norm(h_v)))
        if idx % 1e4 == 0: print(f"{idx}/{num_edges}")

    return res


if __name__ == '__main__':
    cora_hidden_state_path = './outputs/2022-04-14/00-55-02/epoch=350.pkl'
    cora_hidden_state = pickle.load(open(cora_hidden_state_path, 'rb')).detach().cpu().numpy()
    uv_list_path = '../../data/neo_converted/uv_list.pkl'
    uv_dataset = pickle.load(open(uv_list_path, 'rb'))

    X_train = generate_array(cora_hidden_state, uv_dataset['train_u'], uv_dataset['train_v'])
    X_dev = generate_array(cora_hidden_state, uv_dataset['dev_u'], uv_dataset['dev_v'])
    X_test = generate_array(cora_hidden_state, uv_dataset['test_u'], uv_dataset['test_v'])

    features = {
        '__text__': 'graphsage_essay_features',
        'uv_chksum': check_md5(uv_list_path),
        'X_train': X_train,
        'Y_train': uv_dataset['train_y'],
        'X_dev': X_dev,
        'Y_dev': uv_dataset['dev_y'],
        'X_test': X_test,
    }

    clf = fit_lr_classifier(X_train, uv_dataset['train_y'], X_dev, uv_dataset['dev_y'], max_iter=200)

    with open('./graphsage_essay_features.pkl', 'wb') as f:
        pickle.dump(features, f)