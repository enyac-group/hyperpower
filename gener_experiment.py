import argparse
import sys
import string
from copy import copy
import subprocess
import os
import cPickle
from datetime import datetime
import re
import json

SPEARMINT_ROOT = '/home/enyac-awa-r5/research/HyperParameterOptimization/Spearmint-master-repo'  # without the trailing slash
MONGODB_BIN = '/usr/bin/mongod'

def parse_arguments():
    """
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description='Hyper-parameter search with Spearming for a Keras NN model.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment dir')
    parser.add_argument('--optimize', type=str, required=True, help='Metric to optimize: error, energy, runtime, power')
    parser.add_argument('--constraint', type=str, required=False, help='Constraint: error, energy, runtime, power')
    parser.add_argument('--constraint_val', type=str, required=False, help='Constraint value')
    parser.add_argument('--epochs', type=str, required=False, help='Number of Keras training epochs', default='50')
    return parser.parse_args()


def prepare_exp_dir(args):
    """
    Make sure that the experiment subfolders are properly set and that Spearmint is available
    :return:
    """

    # make sure that the experiment subfolders are properly set
    items_to_create = ['/keras_model', '/mongodb', '/spearmint', '/tmp']
    for path in items_to_create:
        if not os.path.exists(args.experiment + path):
            os.mkdir(path)
            print 'Creating %s' % path

    # make sure that Spearmint is available
    error_msgs = ['Spearmint main script was not found at %s. Set the SPEARMINT_ROOT variable.',
                  'Spearmint cleanup script was not found at %s. Set the SPEARMINT_ROOT variable.',
                  'MongoDB is not installed? Server binary not found at %s.',
                  'Your Caffe network description file is needed at %s.']
    items_required = [SPEARMINT_ROOT + '/spearmint/main.py',
                     SPEARMINT_ROOT + '/spearmint/cleanup.sh',
                     MONGODB_BIN, args.experiment + '/keras_model/network_def.py']

    for i, path in enumerate(items_required):
        msg = error_msgs[i]
        if not os.path.exists(path):
            print msg % path
            exit()

    # clean-up previous run
    print 'Cleaning-up previous run ...'
    if os.path.exists(args.experiment + 'spearmint/config.json'):
        subprocess.call('bash ' + SPEARMINT_ROOT + '/spearmint/cleanup.sh' + ' ' +
                        args.experiment + '/spearmint', shell=True)
    subprocess.call('rm -r ' + args.experiment + '/spearmint/*', shell=True)
    subprocess.call('rm -r ' + args.experiment + '/tmp/*', shell=True)


def hyperpower_params(args):
    """
    Save the problem definition (objective function, constraint values, etc.)
    to a pickle file to be used next from the hyperpower.py code issued by
    Spearmint at each iteration.
    :return:
    """
    # store genral parameters for the future use
    hyperpowerparams = {}
    hyperpowerparams['SPEARMINT_ROOT'] = SPEARMINT_ROOT
    hyperpowerparams['optimize'] = args.optimize
    hyperpowerparams['experiment'] = args.experiment
    hyperpowerparams['epochs'] = args.epochs

    if hyperpowerparams['optimize'] == 'error':
        if args.constraint is not None:
            hyperpowerparams['constraint'] = args.constraint
            hyperpowerparams['exec_mode'] = 'constrained'
        else:
            hyperpowerparams['constraint'] = ''
            hyperpowerparams['exec_mode'] = 'unconstrained'
    else:  # if minimizing HW metric (e.g., power), then it makes sense to use error as constraint
        if args.constraint is not None:
            hyperpowerparams['constraint'] = args.constraint
            hyperpowerparams['exec_mode'] = 'constrained'
        else:
            print "Error: error should be used as constraint if optimizing for HW.. Exiting!!"
            exit()

    if hyperpowerparams['exec_mode'] == 'constrained':
        if args.constraint_val is not None:
            hyperpowerparams['constraint_val'] = args.constraint_val
        else:
            print "Error: Constraint metric defined, but not --constraint_val value set.. Exiting!!"
            exit()

    # make sure that 'nvidia-smi' is available if selected metric is energy or power
    if hyperpowerparams['optimize'] in ['energy', 'power'] or hyperpowerparams['constraint'] in ['energy', 'power']:
        try:
            devnull = open(os.devnull, 'w')
            subprocess.call('nvidia-smi', shell=False, stdout=devnull, stderr=devnull)
        except subprocess.CalledProcessError:
            print "Errors with nvidia-smi?? Is it properly installed"
            exit()
        except OSError:
            print "Error: nvidia-smi (executable) not found!! Is it installed?? Exiting!!"
            exit()

    # store hyperpower parameters
    with open(args.experiment + '/tmp/hyperpowerparams.pkl', 'wb') as f:
        cPickle.dump(hyperpowerparams, f)

    return hyperpowerparams


def spearmint_generate_cfg(prefix, hyperpowerparams, net):
    """
         parse HYPERPARAM tokens in the net definition to create the
       spearmint config.json with the respective variables defined

    :param prefix:
    :param hyperpowerparams:
    :param net:
    :return:
    """

    config_buffer = ''  # config str for json file
    param_cnt = {}  # param cnt
    tokens = {}  # tokens, as defined by user inside keras_model/network_def.py
    params = {}  # dict of params

    exec_mode = hyperpowerparams['exec_mode']

    # create header
    if exec_mode == 'unconstrained':
        config_buffer += '{"language": "PYTHON", "main-file": "mainrun.py", ' \
                       '"experiment-name": "hyperpower-' + prefix + '", "likelihood": "GAUSSIAN", "variables" : {'
    elif exec_mode == 'constrained':
        config_buffer += '{"language": "PYTHON", "main-file": "mainrun.py", ' \
                       '"experiment-name": "hyperpower-' + prefix + '", "variables" : {'
    else:
        print "Unknown execution mode selected.. Exiting!"
        exit()

    # parse each token and add it to the Spearmint config file object
    pattern = re.compile('.*HYPERPARAM.*')
    matches = re.findall(pattern, net)
    if len(matches) == 0:
        print "Error: No hyper-parameters!! Make sure you define them in network_def.txt. Exiting!!"
        exit()

    for match in matches:
        (name, param) = match.split('HYPERPARAM')  # extract name and the parameter description
        param_dict = json.loads(param)
        token_name = param_dict['token']  # token entry needed (!!), where the hyper-param name is defined

        # make sure you have not seen this name before !!
        if token_name in params.keys():
            print "Same token name used in multiple hyper-parameter definitions.. Exiting!!"
            exit()

        tokens[len(tokens) + 1] = {'name': token_name, 'description': param}  # store the token
        params[token_name] = param_dict  # store the parsed parameter

        # fill the json file buffer with variable descriptions
        if param_dict['type'] == 'INT':
            config_buffer += '"%s": { "type": "INT", "size": 1, "min": %d, "max": %d},' \
                             % (token_name, param_dict['min'], param_dict['max'])
        if param_dict['type'] == 'FLOAT':
            config_buffer += '"%s": { "type": "FLOAT", "size": 1, "min": %f, "max": %f},' \
                             % (token_name, param_dict['min'], param_dict['max'])
        # if param_dict['type'] == 'ENUM':
        #     config_buffer += '"%s": { "type": "ENUM", "size": 1, "options" : [%s] },' \
        #                      % (token_name, ', '.join([str(x) for x in param_dict['options']]))

    if exec_mode == 'constrained':
        # Make sure you add constraints definition in json
        optimize = hyperpowerparams['optimize']
        constraint = hyperpowerparams['constraint']
        config_buffer = config_buffer[:-1]
        config_buffer += '}, "tasks": {"' + str(optimize) + \
                       '" : {"type" : "OBJECTIVE", "likelihood" : "GAUSSIAN"}, ' \
                       '"' + str(constraint) + '" : {"type" : "CONSTRAINT", "likelihood" : "GAUSSIAN"}}}'
    else:
        # remove extra comma in the end
        config_buffer = config_buffer[:-1]
        config_buffer += '}}'

    # save the json at the experiment path, for Spearmint to use it
    with open(hyperpowerparams['experiment'] + '/spearmint/config.json', 'w') as f:
        f.write(config_buffer)

    # store parsed parameters, for black-box (inner loop) function to use it
    with open(hyperpowerparams['experiment'] + '/tmp/hyperparam_definitions.pkl', 'wb') as f:
        cPickle.dump(params, f)

    return config_buffer, tokens

def spearmint_params(hyperpowerparams):

    netfile = hyperpowerparams['experiment'] + '/keras_model/network_def.py'  # read in Keras definition file
    net = open(netfile, 'r').read()
    tmp_net = copy(net) # templates for keras net file
    prefix = datetime.now().strftime('%Y-%d-%m-%H-%M-%S') # unique prefix for this run

    # parse HYPERPARAM tokens in the net definition to create Spearmint's config.json
    config_buffer, tokens = spearmint_generate_cfg(prefix, hyperpowerparams, net)

    # move hyperpower function to the experiment directory
    subprocess.call('cp hyperpower.py %s/spearmint/mainrun.py' % hyperpowerparams['experiment'], shell=True)

    # generate keras templates
    for i in range(1, len(tokens) + 1):
        # replace HYPERPARAM{...} with HYPERPARAM_name in the template file
        tmp_net = string.replace(tmp_net, tokens[i]['description'], '_' + tokens[i]['name'], 1)
    # store template files
    with open(hyperpowerparams['experiment'] + '/tmp/keras_net_template.py', 'w') as f:
        f.write(tmp_net)


def main(args):

    args = parse_arguments()  # parse run arguments
    prepare_exp_dir(args)  # make sure everything is in place
    hyperpowerparams = hyperpower_params(args)  # set hyperpower arguments
    spearmint_params(hyperpowerparams)  # set spearmint arguments
    subprocess.call("python %s/spearmint/main.py %s/spearmint" %
                    (SPEARMINT_ROOT, args.experiment), shell=True)  # start Spearmint


if __name__ == '__main__':
    main(sys.argv)