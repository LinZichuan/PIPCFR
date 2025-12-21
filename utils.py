import random
import os
import numpy as np
try:
    import torch
except:
    print("Import torch error")
import logging
import yaml
from types import SimpleNamespace


# default config files
DEFAULT_CFG = './configs/default.yaml'
DEFAULT_NEWS_CFG = './configs/NEWS/default.yaml'
DEFAULT_IHDP_CFG = './configs/IHDP/default.yaml'
DEFAULT_SYN_CFG = './configs/synthetic/default.yaml'
DEFAULT_REALWORLD_KNN_CFG = './configs/real_world_knn/default.yaml'
DEFAULT_REALWORLD_PSM_CFG = './configs/real_world_psm/default.yaml'


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_logger(log_name='running_time_log', log_path='filelogs/running_time.log'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    if not os.path.exists("filelogs"):
        os.mkdir("filelogs")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def get_config(cfg_files, params=None):
    """
    Reads a list of YAML configuration files in order, merging them into a single dictionary.
    Values from later files override those from earlier ones.

    Args:
        config_paths (list): A list of file paths to the configuration files. The order is crucial.

    Returns:
        dict: A single, merged configuration dictionary.
              Returns an empty dictionary if no valid files are found.
    """
    if isinstance(cfg_files, str):
        return load_yaml(cfg_files)
    merged_cfg = {}
    for cfg_file in cfg_files:
        if not cfg_file:
            continue
        cfg = load_yaml(cfg_file)
        if cfg:
            merged_cfg.update(cfg)
    if params and isinstance(params, dict):
        merged_cfg.update(params)
    return merged_cfg


def load_yaml(yaml_file):
    if not os.path.exists(yaml_file):
        print(f"Error: File '{yaml_file}' does not exist")
        return None
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as exc:
        print(f"Error: An error occurred while parsing the YAML file: {exc}")
        return None
    except Exception as e:
        print(f"An unknown error occurred: {e}")
        return None

def get_config_dict(model_args):
    cfg_files = [DEFAULT_CFG]
    if model_args["dataset"] == "NEWS":
        cfg_files.append(DEFAULT_NEWS_CFG)
    elif model_args["dataset"] == "synthetic":
        cfg_files.append(DEFAULT_SYN_CFG)
    elif model_args["dataset"] == "IHDP":
        cfg_files.append(DEFAULT_IHDP_CFG)
    elif model_args["dataset"] == "real_world_knn":
        cfg_files.append(DEFAULT_REALWORLD_KNN_CFG)
    elif model_args["dataset"] == "real_world_psm":
        cfg_files.append(DEFAULT_REALWORLD_PSM_CFG)
    else:
        raise ValueError(f"Unknown dataset: {model_args['dataset']}")
    cfg_files.append(model_args["config"])
    if "params" in model_args:
        cfg = get_config(cfg_files, model_args["params"])
    else:
        cfg = get_config(cfg_files)
    cfg["model_name"] = model_args["model_name"]
    return SimpleNamespace(**cfg)


def load_data(fname, feature_util_rate=1.0, next_util_rate=1.0):
    """ Load data set """
    data_in = np.load(fname, allow_pickle=True)
    data_in = {key: data_in[key] for key in data_in.files}
    data = {'x': data_in['x'],
            't': data_in['t'], 
            'yf': data_in['yf'], 
            'ycf': data_in['ycf'],
            'tau': data_in['tau']}
    try:
        data['next_x'] = data_in['next_x']
    except:
        data['next_x'] = None

    try:
        data['num_timesteps'] = data_in['num_timesteps']
        data['m_x'] = data_in['m_x']
        data['m_v'] = data_in['m_v']
        data['m_m'] = data_in['m_m']
        data['m_a'] = data_in['m_a']
    except:
        data['num_timesteps'] = None
        data['m_x'] = None
        data['m_v'] = None
        data['m_m'] = None
        data['m_a'] = None
    
    try:
        data["mu0"] = data_in['mu0']
        data["mu1"] = data_in['mu1']
    except:
        pass

    try:
        data['tau_original'] = data_in['tau_original']
    except:
        data['tau_original'] = data['tau']

    if 'y_scaler' in data_in:
        data['y_scaler'] = data_in['y_scaler'].item()

    if next_util_rate == 0 or (data['num_timesteps'] is not None and data['num_timesteps'] == 0):
        data['next_x'] = np.zeros((data['x'].shape[0], 30, 1))
        data['num_timesteps'] = 0
        data['m_v'] = 10  # 设置为10，保证模型能够正常运行
        data['m_m'] = 10
        data['m_a'] = 10
    elif data['next_x'] is not None and next_util_rate > 0:
        next_x_dim = data['next_x'].shape[1]
        next_x_single_dim = int(data['m_v'] + data['m_m'] + data['m_a'])
        next_x_T = data['num_timesteps']
        used_T = round(next_x_T * next_util_rate)
        next_x_index = []
        next_x_index.extend(np.arange(0, used_T * data['m_v']))
        next_x_index.extend(np.arange(next_x_T * data['m_v'], next_x_T * data['m_v'] + used_T * data['m_m']))
        next_x_index.extend(np.arange(next_x_T * (data['m_v'] + data['m_m']),
                                      next_x_T * (data['m_v'] + data['m_m']) + used_T * data['m_a']))
        data['next_x'] = data['next_x'][:, next_x_index]
        data['num_timesteps'] = used_T

    if feature_util_rate < 1.0:
        feature_use_num = round(data['next_x'].shape[1] * feature_util_rate)
        data['next_x'] = data['next_x'][:, :feature_use_num]
    
    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]
    print('size of short-term variables', data['next_x'].shape, data['num_timesteps'], data['m_v'])
    return data


def sample_data(data, train_ratio=1 / 3, valid_ratio=1 / 3, seed=None, num_samples=None, fix_testset=False):
    num_samples = num_samples if num_samples else data['x'].shape[0]
    valid_size = round(num_samples * valid_ratio)
    test_size = round(num_samples * (1 - train_ratio - valid_ratio))
    train_size = num_samples - valid_size - test_size

    # 使用固定种子保证test dataset不变
    if fix_testset:
        np.random.seed(42)  # 固定种子用于test
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        test_indices = indices[train_size + valid_size:]
        
        # train和valid使用新的种子进行shuffle
        np.random.seed(seed)
        remaining_indices = indices[:train_size + valid_size]
        np.random.shuffle(remaining_indices)
        train_indices = remaining_indices[:train_size]
        valid_indices = remaining_indices[train_size:]
    else:
        np.random.seed(seed)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]

    # 只对数据部分进行采样，排除元数据
    train_data = {key: value[train_indices].squeeze() for key, value in data.items()
                  if isinstance(value, np.ndarray) and value.ndim > 0}
    valid_data = {key: value[valid_indices].squeeze() for key, value in data.items()
                  if isinstance(value, np.ndarray) and value.ndim > 0}
    test_data = {key: value[test_indices].squeeze() for key, value in data.items()
                 if isinstance(value, np.ndarray) and value.ndim > 0}

    # 将元数据直接添加到采样数据中
    for key, value in data.items():
        if not (isinstance(value, np.ndarray) and value.ndim > 0) and key != 'n':
            train_data[key] = value
            valid_data[key] = value
            test_data[key] = value
        elif key == 'n':
            train_data['n'] = train_size
            valid_data['n'] = valid_size
            test_data['n'] = test_size

    return train_data, valid_data, test_data

def convert_next_x_transformer(next_x, data_dict):
    num_timesteps = data_dict['num_timesteps']
    m_v = data_dict['m_v']
    m_m = data_dict['m_m']
    m_a = data_dict['m_a']

    # 处理num_timesteps=0的情况
    if num_timesteps == 0:
        # 返回一个形状为[batch_size, m_v + m_m + m_a]的张量
        return next_x.reshape(next_x.shape[0], m_v + m_m + m_a)

    print("Converting next_x with num_timesteps:", num_timesteps, "m_v:", m_v, "m_m:", m_m, "m_a:", m_a)
    v = next_x[:, :num_timesteps * m_v].reshape(next_x.shape[0], num_timesteps, m_v)
    m = next_x[:, num_timesteps * m_v: num_timesteps * (m_v + m_m)].reshape(next_x.shape[0], num_timesteps, m_m)
    a = next_x[:, num_timesteps * (m_v + m_m): num_timesteps * (m_v + m_m + m_a)].reshape(next_x.shape[0],
                                                                                            num_timesteps, m_a)
    return np.concatenate((v, m, a), axis=2)

def convert_next_x_transformer_realworld(next_x, data_dict):
    num_timesteps = data_dict['num_timesteps']
    m_v = 0
    m_m = 32
    m_a = 0
    num_timesteps = 10

    v = np.zeros((next_x.shape[0], num_timesteps, m_v))
    m = next_x[:, num_timesteps * m_v: num_timesteps * (m_v + m_m)].reshape(next_x.shape[0], m_m, num_timesteps)
    m = np.transpose(m, (0, 2, 1))
    a = np.zeros((next_x.shape[0], num_timesteps, m_a))
    return np.concatenate((v, m, a), axis=2)

    # 处理num_timesteps=0的情况
    # if num_timesteps == 0:
    #     # 返回一个形状为[batch_size, m_v + m_m + m_a]的张量
    #     return next_x.reshape(next_x.shape[0], m_v + m_m + m_a)

    # v = next_x[:, :num_timesteps * m_v].reshape(next_x.shape[0], num_timesteps, m_v)
    # m = next_x[:, num_timesteps * m_v: num_timesteps * (m_v + m_m)].reshape(next_x.shape[0], num_timesteps, m_m)
    # a = next_x[:, num_timesteps * (m_v + m_m): num_timesteps * (m_v + m_m + m_a)].reshape(next_x.shape[0],
    #                                                                                         num_timesteps, m_a)
    # return np.concatenate((v, m, a), axis=2)

def get_sample_weight(t_labels, reweight_sample=None):
    p_t = torch.mean(t_labels).item()
    if reweight_sample:
        w_t = t_labels / (
                2 * p_t)
        w_c = (1 - t_labels) / (2 * (1 - p_t))
        sample_weight = w_t + w_c
    else:
        sample_weight = torch.ones_like(t_labels)
    return sample_weight
