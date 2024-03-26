from .. import config as cfg


# define configurations for convolutions

def init_enc_conv_configs(conv_factor, img_size, enc_dec_layers, pool_layers, start_channels):
    conv_configs = []
    for i in range(enc_dec_layers):
        conv_config = {}
        conv_config['bn'] = True
        if i == 0:
            if cfg.disable_first_bn:
                conv_config['bn'] = False
            conv_config['in_channels'] = start_channels
            conv_config['kernel'] = (7, 7)
        else:
            conv_config['in_channels'] = conv_factor // (2 ** (enc_dec_layers - i))
            if i < enc_dec_layers - 1:
                conv_config['kernel'] = (5, 5)
            else:
                conv_config['kernel'] = (3, 3)
        conv_config['out_channels'] = conv_factor // (2 ** (enc_dec_layers - i - 1))
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = [size // (2 ** i) if size % (2 ** i) == 0 else
                                   size // (2 ** i) + 1 for size in img_size]
        conv_config['rec_size'] = [size // 2 if size % 2 == 0 else size // 2 + 1 for size in conv_config['img_size']]

        conv_configs.append(conv_config)
    for i in range(pool_layers):
        conv_config = {}
        conv_config['bn'] = True
        conv_config['in_channels'] = conv_factor
        conv_config['kernel'] = (3, 3)
        conv_config['out_channels'] = conv_factor
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = [size // (2 ** (enc_dec_layers + i)) if size % (2 ** (enc_dec_layers + i)) == 0
                                   else size // (2 ** (enc_dec_layers + i)) + 1 for size in img_size]
        conv_config['rec_size'] = [size // 2 if size % 2 == 0 else size // 2 + 1 for size in conv_config['img_size']]
        conv_configs.append(conv_config)

    return conv_configs


def init_dec_conv_configs(conv_factor, img_size, enc_dec_layers, pool_layers, start_channels, end_channels):
    conv_configs = []
    for i in range(pool_layers):
        conv_config = {}
        conv_config['bn'] = True
        conv_config['in_channels'] = conv_factor
        conv_config['kernel'] = (3, 3)
        conv_config['out_channels'] = conv_factor
        conv_config['skip_channels'] = cfg.skip_layers * conv_factor
        conv_config['img_size'] = [size // (2 ** (enc_dec_layers + pool_layers - i - 1))
                                   if size % (2 ** (enc_dec_layers + pool_layers - i - 1)) == 0
                                   else size // (2 ** (enc_dec_layers + pool_layers - i - 1)) + 1 for size in img_size]
        conv_config['rec_size'] = [size // 2 if size % 2 == 0 else size // 2 + 1 for size in conv_config['img_size']]
        conv_configs.append(conv_config)
    for i in range(1, enc_dec_layers + 1):
        conv_config = {}
        conv_config['bn'] = True
        conv_config['in_channels'] = conv_factor // (2 ** (i - 1))
        conv_config['kernel'] = (3, 3)
        if i == enc_dec_layers:
            conv_config['out_channels'] = end_channels
            conv_config['skip_channels'] = cfg.skip_layers * start_channels
            conv_config['bn'] = False
        else:
            conv_config['out_channels'] = conv_factor // (2 ** i)
            conv_config['skip_channels'] = cfg.skip_layers * conv_factor // (2 ** i)
        conv_config['img_size'] = [size // (2 ** (enc_dec_layers - i)) if size % (2 ** (enc_dec_layers - i)) == 0
                                   else size // (2 ** (enc_dec_layers - i)) + 1 for size in img_size]
        conv_config['rec_size'] = [size // 2 if size % 2 == 0 else size // 2 + 1 for size in conv_config['img_size']]
        conv_configs.append(conv_config)
    return conv_configs


def init_enc_conv_configs_orig(img_size, enc_dec_layers, start_channels, num_channels):
    conv_configs = []

    conv_config = {}
    conv_config['bn'] = False
    conv_config['in_channels'] = start_channels
    conv_config['kernel'] = (7, 7)
    conv_config['out_channels'] = num_channels
    conv_config['skip_channels'] = 0
    conv_config['img_size'] = img_size
    conv_configs.append(conv_config)

    conv_config = {}
    conv_config['bn'] = True
    conv_config['in_channels'] = num_channels
    conv_config['kernel'] = (5, 5)
    conv_config['out_channels'] = num_channels * 2
    conv_config['skip_channels'] = 0
    conv_config['img_size'] = img_size
    conv_configs.append(conv_config)

    conv_config = {}
    conv_config['bn'] = True
    conv_config['in_channels'] = num_channels * 2
    conv_config['kernel'] = (5, 5)
    conv_config['out_channels'] = num_channels * 4
    conv_config['skip_channels'] = 0
    conv_config['img_size'] = img_size
    conv_configs.append(conv_config)

    if enc_dec_layers > 3:
        conv_config = {}
        conv_config['bn'] = True
        conv_config['in_channels'] = num_channels * 4
        conv_config['kernel'] = (3, 3)
        conv_config['out_channels'] = num_channels * 8
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size
        conv_configs.append(conv_config)

    for i in range(4, enc_dec_layers):
        conv_config = {}
        conv_config['bn'] = True
        conv_config['in_channels'] = num_channels * 8
        conv_config['kernel'] = (3, 3)
        conv_config['out_channels'] = num_channels * 8
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size
        conv_configs.append(conv_config)

    return conv_configs


def init_dec_conv_configs_orig(img_size, enc_dec_layers, end_channels, num_channels):
    conv_configs = []

    for i in range(4, enc_dec_layers):
        conv_config = {}
        conv_config['bn'] = True
        conv_config['in_channels'] = num_channels * 8 + num_channels * 8
        conv_config['kernel'] = (3, 3)
        conv_config['out_channels'] = num_channels * 8
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size
        conv_configs.append(conv_config)

    if enc_dec_layers > 3:
        conv_config = {}
        conv_config['bn'] = True
        conv_config['in_channels'] = num_channels * 8 + num_channels * 4
        conv_config['kernel'] = (3, 3)
        conv_config['out_channels'] = num_channels * 4
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size
        conv_configs.append(conv_config)

    conv_config = {}
    conv_config['bn'] = True
    conv_config['in_channels'] = num_channels * 4 + num_channels * 2
    conv_config['kernel'] = (3, 3)
    conv_config['out_channels'] = num_channels * 2
    conv_config['skip_channels'] = 0
    conv_config['img_size'] = img_size
    conv_configs.append(conv_config)

    conv_config = {}
    conv_config['bn'] = True
    conv_config['in_channels'] = num_channels * 2 + num_channels
    conv_config['kernel'] = (3, 3)
    conv_config['out_channels'] = num_channels
    conv_config['skip_channels'] = 0
    conv_config['img_size'] = img_size
    conv_configs.append(conv_config)

    conv_config = {}
    conv_config['bn'] = False
    conv_config['in_channels'] = num_channels + end_channels
    conv_config['kernel'] = (3, 3)
    conv_config['out_channels'] = end_channels
    conv_config['skip_channels'] = 0
    conv_config['img_size'] = img_size
    conv_configs.append(conv_config)

    return conv_configs
