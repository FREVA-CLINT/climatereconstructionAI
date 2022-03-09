import sys

from .. import config as cfg

# define configurations for convolutions

def init_enc_conv_configs(img_size, enc_dec_layers, pool_layers, start_channels):
    conv_configs = []
    for i in range(enc_dec_layers):
        conv_config = {}
        conv_config['bn'] = True
        if i == 0:
            conv_config['in_channels'] = start_channels
            if cfg.disable_first_bn:
                conv_config['bn'] = False
        else:
            conv_config['in_channels'] = img_size // (2 ** (enc_dec_layers - i))
        conv_config['out_channels'] = img_size // (2 ** (enc_dec_layers - i - 1))
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size // (2 ** i)
        conv_configs.append(conv_config)
    for i in range(pool_layers):
        conv_config = {}
        conv_config['in_channels'] = img_size
        conv_config['out_channels'] = img_size
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size // (2 ** (enc_dec_layers + i))
        conv_configs.append(conv_config)
    return conv_configs

def init_enc_conv_configs_orig(img_size, enc_dec_layers, start_channels):
    conv_configs = []
    same_layers = 3
    for i in range(enc_dec_layers):
        conv_config = {}
        conv_config['bn'] = True
        if i >= same_layers:
            conv_config['in_channels'] = img_size
            conv_config['out_channels'] = img_size
        else:
            if i == 0:
                conv_config['in_channels'] = start_channels
                conv_config['bn'] = False
            else:
                conv_config['in_channels'] = (2**(i-1)) * img_size // 4
            conv_config['out_channels'] = (2**i) * img_size // 4
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size
        conv_configs.append(conv_config)

    return conv_configs

def init_dec_conv_configs(img_size, enc_dec_layers, pool_layers, start_channels, end_channels):
    conv_configs = []
    for i in range(pool_layers):
        conv_config = {}
        conv_config['in_channels'] = img_size
        conv_config['out_channels'] = img_size
        conv_config['skip_channels'] = cfg.skip_layers * img_size
        conv_config['img_size'] = img_size // (2 ** (enc_dec_layers + pool_layers - i - 1))
        conv_configs.append(conv_config)
    for i in range(1, enc_dec_layers + 1):
        conv_config = {}
        conv_config['in_channels'] = img_size // (2 ** (i - 1))
        conv_config['bn'] = True
        if i == enc_dec_layers:
            conv_config['out_channels'] = end_channels
            conv_config['skip_channels'] = cfg.skip_layers * start_channels
            conv_config['bn'] = False
        else:
            conv_config['out_channels'] = img_size // (2 ** i)
            conv_config['skip_channels'] = cfg.skip_layers * img_size // (2 ** i)
        conv_config['img_size'] = img_size // (2 ** (enc_dec_layers - i))
        conv_configs.append(conv_config)
    return conv_configs

def init_dec_conv_configs_orig(img_size, enc_dec_layers, end_channels):
    conv_configs = []
    same_layers = enc_dec_layers-4
    k = 0
    for i in range(enc_dec_layers):
        conv_config = {}
        conv_config['bn'] = True
        if i <= same_layers:
            conv_config['in_channels'] = 2*img_size
            conv_config['out_channels'] = img_size
        else:
            k += 1
            if i == enc_dec_layers -1:
                conv_config['in_channels'] = end_channels+conv_configs[-1]['out_channels']
                conv_config['out_channels'] = end_channels
                conv_config['bn'] = False
            else:
                conv_config['out_channels'] = img_size // (2 ** k)
                conv_config['in_channels'] = 3*conv_config['out_channels']
        conv_config['skip_channels'] = 0
        conv_config['img_size'] = img_size
        conv_configs.append(conv_config)
    return conv_configs
