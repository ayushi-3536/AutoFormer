import yaml

def _update_config(base_cfg, exp_cfg):
    for k, v in exp_cfg.items():
        base_cfg[k] = v
    # print(base_cfg)
    return base_cfg


def update_config_from_file(cfg, filename):
    exp_config = None
    with open(filename) as f:
        exp_config = dict(yaml.safe_load(f))
        new_cfg =_update_config(cfg, exp_config)
    return new_cfg


