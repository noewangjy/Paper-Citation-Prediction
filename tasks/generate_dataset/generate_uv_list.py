import os
import pickle
import shutil


def check_md5(path, display: bool = True):
    res = os.popen(f'md5sum {path}').readlines()[0].split(' ')[0]
    if display:
        print(f"MD5SUM of {path}: {res}")
    return res


if __name__ == '__main__':
    whole_dataset_path = '../../data/neo_converted/nullptr_no_feature_whole.pkl'
    train_dataset_path = '../../data/neo_converted/nullptr_no_feature_train.pkl'
    dev_dataset_path = '../../data/neo_converted/nullptr_no_feature_dev.pkl'
    test_dataset_path = '../../data/neo_converted/nullptr_no_feature_test.pkl'

    print(check_md5(whole_dataset_path))
    print(check_md5(train_dataset_path))
    print(check_md5(dev_dataset_path))
    print(check_md5(test_dataset_path))

    whole_dataset = pickle.load(open(whole_dataset_path, 'rb'))
    train_dataset = pickle.load(open(train_dataset_path, 'rb'))
    dev_dataset = pickle.load(open(dev_dataset_path, 'rb'))
    test_dataset = pickle.load(open(test_dataset_path, 'rb'))

    res = {
        'train_u': train_dataset['u'],
        'train_v': train_dataset['v'],
        'train_y': train_dataset['y'],
        'dev_u': dev_dataset['u'],
        'dev_v': dev_dataset['v'],
        'dev_y': dev_dataset['y'],
        'test_u': test_dataset['u'],
        'test_v': test_dataset['v']
    }

    with open('./uv_list.pkl', 'wb') as f:
        pickle.dump(res, f)

    print(check_md5('./uv_list.pkl'))

    with open('./uv_list.chksum', 'w') as f:
        f.write(check_md5('./uv_list.pkl'))

    shutil.copy('./uv_list.chksum', '../../data/converted')
    shutil.copy('./uv_list.pkl', '../../data/converted')
    shutil.copy('./uv_list.chksum', '../../data/neo_converted')
    shutil.copy('./uv_list.pkl', '../../data/neo_converted')