import os
import sys
import pathlib
import argparse
import configparser
import numpy as np

from pathlib import Path

from collections import OrderedDict
from xfusion import log
from distutils.util import strtobool

__author__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['config_to_list',
           'get_config_name',
           'log_values',
           'parse_known_args',
           'write']

CONFIG_FILE_NAME = os.path.join(str(pathlib.Path.home()), 'xfusion.conf')
XFUSION_HOME = os.path.join(str(pathlib.Path.home()), 'xfusion')
XFUSION_TRAIN_HOME = os.path.join(str(XFUSION_HOME), 'train')
XFUSION_INFERENCE_HOME = os.path.join(str(XFUSION_HOME), 'inference')
XFUSION_LOG_HOME = os.path.join(str(XFUSION_HOME), 'log')

SECTIONS = OrderedDict()

SECTIONS['general'] = {
    'config': {
        'default': CONFIG_FILE_NAME,
        'type': str,
        'help': "File name of configuration",
        'metavar': 'FILE'},
    'model_type': {
        'default': 'SwinIRModel',
        'choices': ['SwinIRModel','EDVRModel'],
        'type': str,
        'help': 'Model type'
    },
    'verbose': {
        'default': True,
        'help': 'Verbose output',
        'action': 'store_true'},
        }

SECTIONS['home'] = {
    'home': {
        'default': XFUSION_HOME,
        'type': Path,
        'help': 'name of the home directory for the output files',
        'metavar': 'FILE'},
    'train-home': {
        'default': XFUSION_TRAIN_HOME,
        'type': Path,
        'help': 'name of the home directory for the output files',
        'metavar': 'FILE'},
    'inference-home': {
        'default': XFUSION_INFERENCE_HOME,
        'type': Path,
        'help': 'name of the home directory for the inference files',
        'metavar': 'FILE'},
    'log-home': {
        'default': XFUSION_LOG_HOME,
        'type': Path,
        'help': 'name of the home directory for the log files',
        'metavar': 'FILE'},
}

SECTIONS['convert'] = {
    'dir-lo-convert': {
        'default': os.path.join(XFUSION_TRAIN_HOME,"train_sharp_bicubic/X4/"),
        'type': Path,
        'help': 'name of the directory with the low resolution images',
        'metavar': 'FILE'},
    'dir-hi-convert': {
        'default': os.path.join(XFUSION_TRAIN_HOME,"train_sharp/"),
        'type': Path,
        'help': 'name of the directory with the high resolution images',
        'metavar': 'FILE'},
    'out-dir-lo': {
        'default': os.path.join(XFUSION_TRAIN_HOME,"train_sharp_mono_1ch_bicubic/X4/"),
        'type': Path,
        'help': 'name of the output directory for the low resolution images',
        'metavar': 'FILE'},
    'out-dir-hi': {
        'default': os.path.join(XFUSION_TRAIN_HOME,"train_sharp_mono_1ch/"),
        'type': Path,
        'help': 'name of the output directory for the high resolution images',
        'metavar': 'FILE'},
    }


SECTIONS['train'] = {
    'dir-lo-train': {
        'default': '.',
        'type': Path,
        'help': 'name of the directory with the low resolution images',
        'metavar': 'FILE'},
    'dir-hi-train': {
        'default': '.',
        'type': Path,
        'help': 'name of the directory with the high resolution images',
        'metavar': 'FILE'},
    'dir-lo-val': {
        'default': '.',
        'type': Path,
        'help': 'name of the directory with the low resolution images',
        'metavar': 'FILE'},
    'dir-hi-val': {
        'default': '.',
        'type': Path,
        'help': 'name of the directory with the high resolution images',
        'metavar': 'FILE'},
    'opt' : {
        'default' : '.',
        'type': str,
        'help': "Path to option YAML file."},
    'path-train-meta-info-file': {
        'default': ".",
        'type': Path,
        'help': 'name of the path to training image meta data',
        'metavar': 'FILE'},
    'path-val-meta-info-file': {
        'default': ".",
        'type': Path,
        'help': 'name of the path to validation image meta data',
        'metavar': 'FILE'},
    'pretrain_network_g': {
        'default': 'none',
        'help': "When set continue training from the specified model file"},
    'launcher' : {
        'default' : 'none',
        'choices' : ['none', 'pytorch', 'slurm','polaris'],
        'help': "Job launcher."},
    'auto-resume': {
        'default': False,
        'help': "When set auto-resume is True",
        'action': 'store_true'},
     'local-rank' : {
        'default' : 0,
        'type': int,
        'help': "Local rank."},
    'force-yml': {
        'default': 'none',
        'help': "When set used the yml config file"},
    'debug': {
        'default': False,
        'help': "When set debug is True",
        'action': 'store_true'},
    'is-train': {
        'default': True,
        'help': "When set train is True",
        'action': 'store_true'},    
  }

MODEL_TYPES = {'train':OrderedDict(),'inference':OrderedDict()}
MODEL_TYPES['train']['SwinIRModel'] = {'drop_path_rate': {
            'default': 0.2,
            'type': float,
            'help': "Probability of stochastic depth"},
        'embed_dim': {
            'default': 192,
            'type': int,
            'help': "Dimension of the embedding layer"},
        'scale_depth': {
            'default': 1,
            'type': int,
            'help': "Multiplier of the original model depth"},
        'scale_mha': {
            'default': 1,
            'type': float,
            'help': "Multiplier of the original model transformer block number of attention heads"},
        'window_size_spatial': {
            'default': 8,
            'type': int,
            'help': "Window size to group tokens of vit"},
        'resi_connection': {
            'default': '1conv',
            'choices': ['1conv','3conv','0conv'],
            'type': str,
            'help': "Type of operation in the vit residual connection"},
        'conv_window_size': {
            'default': 3,
            'type': int,
            'help': "Size of convolution window if used in the vit residual connection"},
        # newly added ddp configs
        'iter_div_factor': {
            'default': 10,
            'type': int,
            'help': "Divisor of the total number of iterations"},
        'num_frame': {
            'default': 3,
            'type': int,
            'help': "Number of input low resolution frames to the model"},
        'batch_size': {
            'default': 1,
            'type': int,
            'help': "Number of samples in a batch"},
        'initial_lr': {
            'default': 2e-4,
            'type': float,
            'help': "Initial learning rate"},
        'num_workers': {
            'default': 3,
            'type': int,
            'help': "Number of workers for data loader"},
        'warmup_iter': {
            'default': -1,
            'type': int,
            'help': "Number of iterations for training warm up"},
        'dataset_enlarge_ratio': {
            'default': 10.0,
            'type': float,
            'help': "Multiplier of the original number of training samples"},
        'sub_dataset': {
            'default': False,
            'type': lambda x: bool(strtobool(x)),
            'help': "When set the original training data are randomly sub-sampled for training"},
        'sub_dataset_frac': {
            'default': 1.0,
            'type': float,
            'help': "Fraction of the original training data set size to sub-sample"},
        'single_epoch_ok': {
            'default': False,
            'type': lambda x: bool(strtobool(x)),
            'help': "If to only train data for one epoch"},
        'profiler_iter_num': {
            'default': 3,
            'type': int,
            'help': "Number of subsequent iterations to track computation performance for HTA"},}

MODEL_TYPES['train']['EDVRModel'] = {}


SECTIONS['download'] = {
    'dir-inf': {
        'default': "https://g-a0400.fd635.8443.data.globus.org/rad_00001/rad_00001.zip",
        'type': str,
        'help': 'name of the directory with the images for inference',
        'metavar': 'FILE'},
    'out-dir-inf': {
        'default': XFUSION_INFERENCE_HOME,
        'type': Path,
        'help': 'name of the output directory for the low resolution images',
        'metavar': 'FILE'},
}

SECTIONS['inference'] = {
    'opt' : {
        'default' : '.',
        'type': str,
        'help': "Path to inference option YAML file."},
    'arch-opt': {
        'default': '.',
        'type': str,
        'help': 'Path to train option YAML file.'
    },
    'lo-frame-sep' : {
        'default' : 1,
        'type': int,
        'help': "Low frame sep."},
    'hi-frame-sep' : {
        'default' : 1,
        'type': int,
        'help': "High frame sep."},
    'b0': {
        'default': False,
        'help': "When set debug is True",
        'action': 'store_true'},
    'img-class' : {
        'default' : 'dataset1',
        'type': str,
        'help': "Image class."},
    'infer-mode' : {
        'default' : 'stf',
        'type': str,
        'choices': ['stf','sr','bil','bic','itpf'],
        'help': "Mode"},
    'model_file': {
        'default': 'none',
        'type': Path,
        'help': "When set the path to the model weights to initialize the model"},
    'num_frame': {
        'default': 3,
        'type': int,
        'help': "Number of input low resolution frames to the model"},
    'machine': {
        'default': 'polaris',
        'choices' : ['none', 'tomo','polaris'],
        'help': "machine to run the inference job"
    },
  }

MODEL_TYPES['inference']['SwinIRModel'] = {
        'align_features_ok': {
            'default': False,
            'type': lambda x: bool(strtobool(x)),
            'help': "When set applies deformable conv to align features before vit feature enhancement"},
        'adapt_deformable_conv':{
            'default': False,
            'type': lambda x: bool(strtobool(x)),
            'help': "When set applies linear transformation to match channel size of the deformable conv layer and vit"},
        'scale_mha': {
            'default': 1,
            'type': float,
            'help': "Multiplier of the original model transformer block number of attention heads"},
        'embed_dim': {
            'default': 192,
            'type': int,
            'help': "Dimension of the embedding layer"},
        'window_size_spatial': {
            'default': 8,
            'type': int,
            'help': "Window size to group tokens of vit"},
        'scale_depth': {
            'default': 1,
            'type': int,
            'help': "Multiplier of the original model depth"},}
MODEL_TYPES['inference']['EDVRModel'] = {}

HOME_PARAMS = ('home', )
CONVERT_PARAMS   = ('convert', )
TRAIN_PARAMS     = ('train', )
INFERENCE_PARAMS = ('inference', )
DOWNLOAD_PARAMS = ('download', )
XFUSION_PARAMS   = ('convert', 'home', 'train', 'inference', 'download')

NICE_NAMES = ('General', 'Home', 'Convert', 'Train', 'Download', 'Inference')

def make_default_home_dir():
    SECTIONS['home']['home']['default'].mkdir(exist_ok=True, parents=True)

def make_default_train_home_dir():
    SECTIONS['home']['train-home']['default'].mkdir(exist_ok=True, parents=True)

def make_default_inference_home_dir():
    SECTIONS['home']['inference-home']['default'].mkdir(exist_ok=True, parents=True)

def make_default_log_home_dir():
    SECTIONS['home']['log-home']['default'].mkdir(exist_ok=True, parents=True)

def get_config_name():
    """Get the command line --config option."""
    name = CONFIG_FILE_NAME
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--config'):
            if arg == '--config':
                return sys.argv[i + 1]
            else:
                name = sys.argv[i].split('--config')[1]
                if name[0] == '=':
                    name = name[1:]
                return name

    return name

def get_train_dirs():
    config_name = get_config_name()
    config = configparser.ConfigParser()
    config.read([config_name])
    return config['home']['train-home']

def get_base_log_dirs():
    config_name = get_config_name()
    config = configparser.ConfigParser()
    config.read([config_name])
    return config['home']['log-home']

def get_inf_data_dirs(dataset):
    config_name = get_config_name()
    config = configparser.ConfigParser()
    config.read([config_name])
    return os.path.join(config['home']['inference-home'],dataset)

def get_model_type():
    config_name = get_config_name()
    config = configparser.ConfigParser()
    config.read([config_name])
    return config['general']['model_type']

def parse_known_args(parser, model_type, subparser=False):
    """
    Parse arguments from file and then override by the ones specified on the
    command line. Use *parser* for parsing and is *subparser* is True take into
    account that there is a value on the command line specifying the subparser.
    """
    if len(sys.argv) > 1:
        subparser_value = [sys.argv[1]] if subparser else []
        config_values = config_to_list(model_type, config_name=get_config_name())
        values = subparser_value + config_values + sys.argv[2:]
        #print(subparser_value, config_values, values)
    else:
        values = ""
    print(f"values before parsing: {values}")
    parsed = parser.parse_known_args(values)
    print(f"values after parsing: {parsed}")
    return parsed[0]


def config_to_list(model_type, config_name=CONFIG_FILE_NAME):
    """
    Read arguments from config file and convert them to a list of keys and
    values as sys.argv does when they are specified on the command line.
    *config_name* is the file name of the config file.
    """
    result = []
    config = configparser.ConfigParser()

    if not config.read([config_name]):
        return []

    for section in SECTIONS:
        for name, opts in ((n, o) for n, o in SECTIONS[section].items() if config.has_option(section, n)):
            value = config.get(section, name)

            if value != '' and value != 'None':
                action = opts.get('action', None)

                if action == 'store_true' and value == 'True':
                    # Only the key is on the command line for this action
                    result.append('--{}'.format(name))

                if not action == 'store_true':
                    if opts.get('nargs', None) == '+':
                        result.append('--{}'.format(name))
                        result.extend((v.strip() for v in value.split(',')))
                    else:
                        result.append('--{}={}'.format(name, value))
        
        if section in ['train','inference']:
            for name, opts in ((n, o) for n, o in MODEL_TYPES[section][model_type].items() if config.has_option(section, n)):
                value = config.get(section, name)

                if value != '' and value != 'None':
                    action = opts.get('action', None)

                    if action == 'store_true' and value == 'True':
                        # Only the key is on the command line for this action
                        result.append('--{}'.format(name))

                    if not action == 'store_true':
                        if opts.get('nargs', None) == '+':
                            result.append('--{}'.format(name))
                            result.extend((v.strip() for v in value.split(',')))
                        else:
                            result.append('--{}={}'.format(name, value))

    return result


class Params(object):
    def __init__(self, model_type, sections=()):
        self.sections = sections + ('general',)
        self.model_type = model_type if model_type is not None else SECTIONS['general']['model_type']['default']

    def add_parser_args(self, parser):
        for section in self.sections:
            for name in sorted(SECTIONS[section]):
                opts = SECTIONS[section][name]
                parser.add_argument('--{}'.format(name), **opts)
            
            if section in ['train','inference']:
                for name in MODEL_TYPES[section][self.model_type]:
                    opts = MODEL_TYPES[section][self.model_type][name]
                    parser.add_argument('--{}'.format(name), **opts)

    def add_arguments(self, parser):
        self.add_parser_args(parser)
        return parser

    def get_defaults(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)

        return parser.parse_args('')


def write(config_file, model_type, args=None, sections=None):
    """
    Write *config_file* with values from *args* if they are specified,
    otherwise use the defaults. If *sections* are specified, write values from
    *args* only to those sections, use the defaults on the remaining ones.
    """
    if model_type == None:
        model_type = SECTIONS['general']['model_type']['default']
    config = configparser.ConfigParser()

    for section in SECTIONS:
        config.add_section(section)
        for name, opts in SECTIONS[section].items():
            if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                value = getattr(args, name.replace('-', '_'))
                if isinstance(value, list):
                    print(type(value), value)
                    value = ', '.join(value)
            else:
                value = opts['default'] if opts['default'] is not None else ''

            prefix = '# ' if value == '' else ''

            if name != 'config':
                if name == 'model_type':
                    value = model_type
                config.set(section, prefix + name, str(value))
        
        if section in ['train','inference']:
            for name, opts in MODEL_TYPES[section][model_type].items():
                if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                    value = getattr(args, name.replace('-', '_'))
                    if isinstance(value, list):
                        print(type(value), value)
                        value = ', '.join(value)
                else:
                    value = opts['default'] if opts['default'] is not None else ''

                prefix = '# ' if value == '' else ''

                if name != 'config':
                    config.set(section, prefix + name, str(value))

    with open(config_file, 'w') as f:
        config.write(f)


def log_values(args, model_type):
    """Log all values set in the args namespace.

    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted((k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))
        #log.info(f"entries are: {entries}")
        if entries:
            log.info(name)

            for entry in entries:
                value = args[entry] if args[entry] is not None else "-"
                log.info("  {:<16} {}".format(entry, value))
        
        if name in ['train','inference']:
            entries = sorted((k for k in args.keys() if k.replace('_', '-') in MODEL_TYPES[name][model_type]))
            if entries:
                for entry in entries:
                    value = args[entry] if args[entry] is not None else "-"
                    log.info("  {:<16} {}".format(entry, value)) 


def yaml_args(args, yaml_file, sample, cli_args=sys.argv):
    """Override config parameters on a per-sample basis.
    
    This can be used when processing many tomograms that differ by
    only one or two parameters. These parameters can be saved in a
    yaml file then loaded for the corresponding tomogram without
    affecting the base parameters.
    
    The filenames listed in the YAML file can be relative to the
    current working directory, including subdirectories, but cannot
    use other file-system shortcuts (e.g. “..”, “~”).
    
    Use::
    
        args = ...
        for filename in all_filenames:
            my_args = yaml_args(args, sample=filename)
            recon.rec(my_args)
    
    The yaml file is expected to be in the following example format,
    where some first level entry should match the *sample* argument::
    
        tomo_file_1.h5:
          rotation_axis: 512
          remove_stripe_method: ti
        tomo_file_2.h5:
          remove_stripe_method: none
    
    Parameters
    ==========
    args
      The base-line configuration args to be copied and modified.
    yaml_file
      The path to the a yaml file with overridden parameters.
    sample
      The name of the sample to find in the yaml file. Most likely to
      be the name of the HDF5 file.
    cli_args
      A list of CLI parameters, similar to ``sys.argv``. Any
      parameters in this list will not be overridden by the yaml file.
    
    Returns
    =======
    new_args
      A copy of *args* with new parameters based on what was found in
      the yaml file *yaml_file*.

    """
    sample = Path(sample)
    yaml_file = Path(yaml_file)
    # Check for bad files
    if not yaml_file.exists():
        log.warning("  *** YAML file does not exist: %s", yaml_file)
        return args
    # Look for the requested key in a hierarchical manner
    with open(yaml_file, mode='r') as fp:
        extra_params = None
        yaml_data = yaml.safe_load(fp)
        if yaml_data is None:
            log.warning("  *** Invalid YAML file: %s", yaml_file)
            return args
        keys_to_check = [sample] + [sample.relative_to(a) for a in sample.parents]
        for key in keys_to_check:
            key = str(key)
            if key in yaml_data.keys():
                extra_params = yaml_data[key]
                log.debug("  *** Found %d extra parameters for %s", len(extra_params), key)
                break
        if extra_params is None:
            raise KeyError(sample)
    # Create a copy of the args
    new_args = copy(args)
    # Prepare CLI parameters by only keep the "--arg" part
    cli_args = [p.split('=')[0] for p in cli_args]
    # Update with new values
    new_args.file_name = Path(sample)
    for key, value in extra_params.items():
        params_key = "--{}".format(key.replace('_', '-'))
        # Parameters given on the command line take precedence
        is_in_cli = len([p for p in cli_args if p == params_key]) > 0
        if not is_in_cli:
            setattr(new_args, key.replace('-', '_'), value)
    # Return the modified parameters
    return new_args