import yaml
import logging
from easydict import EasyDict


def setup_logger(env="debug", save_logs=False):
    logger = logging.getLogger(__name__)
    if env == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if save_logs:  # TODO: test functionality
        file_handler = logging.FileHandler('example.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_config_dict(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)
    return config


def get_data_config(data_config, test_config):
    # data_config = get_config_dict(path)
    data_config.config.sampling_strategy = test_config.sampling_type
    return data_config


def get_model_config(model_config, data_config):
    # model_config = get_config_dict(path)
    model_config = model_config.config

    model_config.encoder.hist_len = data_config.config.hist_len
    model_config.encoder.pred_lens = data_config.config.pred_lens

    model_config.encoder.T_size = data_config.config.traj_len
    model_config.encoder.A_size = data_config.config.k_agents
    model_config.encoder.interp_flag = data_config.config.encode_interp_flag

    encoder_type = model_config.encoder.context_encoder_type
    encoder_type = f"contextnet_{encoder_type}"

    model_config.encoder.contextnet.embed_size = model_config.encoder.embed_size
    model_config.encoder.contextnet.num_vectors = data_config.config.num_polylines

    model_config['encoder'][encoder_type]['num_vectors'] = data_config['config']['num_polylines']
    model_config['encoder'][encoder_type]['embed_size'] = model_config['encoder']['embed_size']

    return model_config
