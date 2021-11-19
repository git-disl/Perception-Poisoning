import os


def create_output_dir(config):
    components = [config.root_dir_output, config.dataset]

    if config.benign:
        components += ['benign']
    elif config.poison == 'objn':
        components += ['poison-%s-%s' % (config.poison, config.source_class)]
    elif config.poison == 'bbox':
        components += ['poison-%s_%.2f-%s' % (config.poison, config.bbox_shrinkage, config.source_class)]
    elif config.poison == 'class':
        components += ['poison-%s-%s-%s' % (config.poison, config.source_class, config.target_class)]

    o = []
    o += ['clients-%d' % config.num_clients]
    if not config.benign:
        o += ['malicious-%d' % (config.num_clients * config.frac_malicious)]
    o += ['participants-%d' % (config.num_clients * config.frac_participants)]
    if not config.benign and config.alpha is not None:
        o += ['alpha-%.2f' % config.alpha]
    o += ['epochs-%d' % config.local_epochs]
    o += ['%s' % config.optimizer]
    components += ['_'.join(o)]

    path = os.path.join(*components)
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'ckpt'))
        os.makedirs(os.path.join(path, 'grad'))
        os.makedirs(os.path.join(path, 'eval'))
    return path
