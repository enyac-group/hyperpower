import numpy as np
import cPickle, json
import math, os, sys
import string
import re, time
import subprocess
from datetime import datetime


def generate_keras_executable(hyperparam_definitions, params, prefix):
    """
        Generate the Keras executable by replacing the replace hyper-params
    definitions with their instance values.

    :param hyperparam_definitions: the hyper-params definitios
    :param params: the (hyper)parameter values for the current
                   instance suggested by Spearmint
    :return:
    """

    # transform parameters according to transformation specified in the model file
    for p in params:
        if hyperparam_definitions[p].get('transform', None) is not None:

            # X<>: multiplier where <> stands for any number (examples: X10, X100, X22)
            if hyperparam_definitions[p]['transform'][0] == 'X':
                multiplier = int(hyperparam_definitions[p]['transform'][1:])
                params[p][0] *= multiplier

            # LOG<>: number which goes to Spearmint corresponds to log with base <> of an actual
            #        number (example: value 2 of LOG10 corresponds to 100)
            if hyperparam_definitions[p]['transform'][0:3] == 'LOG':
                base = int(hyperparam_definitions[p]['transform'][3:])
                params[p][0] = math.log(params[p][0], base)

            # NEGEXP<>: where <> is  the base, the number which goes to Spearmint is negative of the
            #           exponent (example: value 3 with NEGEXP10 means 10^-3 and correpsonds to 0.001)
            if hyperparam_definitions[p]['transform'][0:6] == 'NEGEXP':
                negexp = float(hyperparam_definitions[p]['transform'][6:])
                params[p] = [negexp ** float(-params[p][0])]

    print params

    # generate Keras executable file with the current set of paramters
    tmp_net = open('../tmp/keras_net_template.py', 'r').read()
    for p in params:
        tmp_net = string.replace(tmp_net, 'HYPERPARAM_' + p, str(params[p][0]), 1)
    # store it for future reference
    with open('../tmp/%s_keras_net.py' % prefix, 'w') as f:
        f.write(tmp_net)

    return tmp_net


def execute_keras(keras_net, epochs):

    model = None
    x_train, y_train, batch_size, x_test, y_test = None, None, None, None, None
    exec keras_net

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=0, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print 'Test loss:', score[0]
    print 'Test accuracy:', score[1]

    return float(score[0]), float(score[1]), history


def profile_keras(keras_net):

    model = None
    x_train, y_train, batch_size, x_test, y_test = None, None, None, None, None
    exec keras_net

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1,
                        verbose=0, validation_data=(x_test, y_test))

    batch_size = 100
    num_measurements = 30
    power_measurements = []
    runtime_measurements = []
    energy_measurements = []
    print "Starting nvidia @", datetime.now().strftime('%Y-%d-%m-%H-%M-%S')
    p = subprocess.Popen(["nvidia-smi", "-i", "0", "-lms", "1", "-q",
                          "-d", "POWER", "-f", "../tmp/nvidia-smi-log.txt"], shell=False)
    # iterate multiple times to get a representative value
    for _ in range(num_measurements):

        start_time_meas = time.time()
        time.sleep(1)
        score = model.evaluate(x_test[0:batch_size], y_test[0:batch_size], verbose=0)
        elapsed_time_meas = time.time() - start_time_meas
        runtime_measurements.append(elapsed_time_meas)

    p.kill()
    print "Killing nvidia @", datetime.now().strftime('%Y-%d-%m-%H-%M-%S')

    # parse the output file of nvidia-smi
    logbuffer = open("../tmp/nvidia-smi-log.txt", 'r').read()
    pattern = re.compile("Avg\s+: [0-9]+\.[0-9]+ W")  # find optimization tokens in the buffer
    matches = re.findall(pattern, logbuffer)

    if len(matches) == 0:
        print "Error: No nvidia-smi values found in log file... Exiting!!"
        exit()

    # parse each token and add it to the Spearmint config file object
    values = []
    for match in matches:
        val = float(match.split(':')[1].split('W')[0].strip())
        values.append(val)

    power = np.mean(values)
    energy = power * np.mean(runtime_measurements)

    return np.mean(runtime_measurements), power, energy


def keras_run(params):

    prefix = datetime.now().strftime('%Y-%d-%m-%H-%M-%S')  # unique prefix for this run
    start_time = time.time()
    print "Initial time stamp: ", start_time

    # load general and optimization parameters
    with open('../tmp/hyperparam_definitions.pkl', 'rb') as f:
        hyperparam_definitions = cPickle.load(f)
    with open('../tmp/hyperpowerparams.pkl', 'rb') as f:
        hyperpowerparams = cPickle.load(f)
    optimize = hyperpowerparams['optimize']
    experiment = hyperpowerparams['experiment']
    epochs = int(hyperpowerparams['epochs'])
    exec_mode = hyperpowerparams['exec_mode']
    objective_current_value, constraint_current_value = None, None
    if exec_mode == 'constrained':
        constraint = hyperpowerparams['constraint']
        constraint_val = float(hyperpowerparams['constraint_val'])

    # generate the Keras executable
    keras_net = generate_keras_executable(hyperparam_definitions, params, prefix)

    if exec_mode == 'unconstrained':
        # Train model to obtain accuracy
        loss, accuracy, history = execute_keras(keras_net, epochs)
        print "Loss", loss, "\nAccuracy", accuracy, "\nHistory", history.history['val_acc']
        accuracy_100 = accuracy * 100.0
        error = 100.0 - float(accuracy_100)  # compute the error that you want to minimize
        elapsed_time = time.time() - start_time
        print "Elapsed time (s): ", elapsed_time
        return error

    elif exec_mode == 'constrained':

        # in constrained case you always have a HW metric, cheaper to evaluate first
        runtime, power, energy = profile_keras(keras_net)
        print runtime, power, energy

        # check if the HW constraint is satisfied, otherwise exit
        if optimize == 'error':

            if constraint == 'energy':
                constraint_current_value = energy
            if constraint == 'power':
                constraint_current_value = power
            if constraint == 'runtime':
                constraint_current_value = runtime

            if constraint_current_value >= constraint_val:
                # HW constraint violated, return NaN for objective (max accuracy)
                elapsed_time = time.time() - start_time
                print "Elapsed time (s): ", elapsed_time
                return {
                    optimize: np.NaN,
                    constraint: constraint_val - constraint_current_value
                }
            else:
                # HW constraint satisfied, train model evaluate accuracy
                loss, accuracy, history = execute_keras(keras_net, epochs)
                print "Loss", loss, "\nAccuracy", accuracy, "\nHistory", history.history['val_acc']
                accuracy_100 = accuracy * 100.0
                error = 100.0 - float(accuracy_100)  # compute the error that you want to minimize
                elapsed_time = time.time() - start_time
                print "Elapsed time (s): ", elapsed_time
                return {
                    optimize: error,
                    constraint: constraint_val - constraint_current_value
                }

        else:

            # HW metric (w.r.t. we minimize) is cost, error constraint is evaluated
            if optimize == 'energy':
                objective_current_value = energy
            if optimize == 'power':
                objective_current_value = power
            if optimize == 'runtime':
                objective_current_value = runtime

            # HW objective computed, train model evaluate accuracy
            loss, accuracy, history = execute_keras(keras_net, epochs)
            print "Loss", loss, "\nAccuracy", accuracy, "\nHistory", history.history['val_acc']
            accuracy_100 = accuracy * 100.0
            error = 100.0 - float(accuracy_100)  # compute the error that you want to minimize
            constraint_current_value = error
            elapsed_time = time.time() - start_time
            print "Elapsed time (s): ", elapsed_time
            return {
                optimize: objective_current_value,
                constraint: constraint_val - constraint_current_value
            }


# Write a function like this called 'main'
def main(job_id, params):
    return keras_run(params)