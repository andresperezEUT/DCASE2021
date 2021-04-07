"""
Optimization heuristic for finding out best combination of localization/detection parameters.
Perform simulated annealing search [1] by joint minimization of two cost functions:
- event precision (rate of false positives) should be 0.6 (40% of instances of undesired class)
- event recall (rate of false negatives) should be 1
Frame-level metrics reported for completion.
[1] How to Implement Simulated Annealing Algorithm in Python.  https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
"""
from utils import create_folder

"""
RAFA: aquí está el script que he hecho para simulated annealing, con el espacio de parámetros discretizado.
Échale un vistazo a la implementación, hay dos cosas que no me convencen
- Los vecinos se cambian todos a la vez, esto no se si es así:
        * Efectivamente este paso es bastante importante y requiere definir una función de vecindad adecuada. Yo siempre he trabajado con los siguientes parámetros:
            - Nvar: Número de variables que cambian en cada iteración. Hay muchas técnicas y lo suyo es calibrarlo con experimientos. Lo pongo en 3 por empezar por algo. Podemos ir cambiándolo
            - Nsaltos: Cuantos saltos damos al cambiar cada variable. Como el espacio de búsqueda que has definido es bastante bajo, lo dejaré en uno
            Estos pueden ser estáticos o dinámicos. La idea es la misma, que los movimientos sean más pequeños conforme vamos avanzando en la búsqueda. Como primera aproximación los dejamos estáticos.
- Cuando los nuevos parámetros no mejoran, la función con la que se compara el random no me convence (l. 200)
        * Aquí te refieres a la función de probabilidad??
        
        *He añadido alguna cosa más: cadenas de markov, factor geométrico en vez del alpha, almacenar mejor solución (que no tiene porque ser la actual)...lo hablamos
        
        
Si quieres ejecutarlo, puedes poner la variable `simulate_cost` a true para no tener que llamar los métodos de verdad.
"""

"""
Resultados preliminares de coste:
- Aprox. 6 segundos de media por analizar 1 archivo de audio
- Probando con 20 archivos por dataset (en vez de 400, reducción x20) son 120 s = 2 minutos por iteración
- Con 2000 iteraciones, salen aprox. 3 días de tiempo de ejecución (60 con el dataset completo)

"""

"""
resultados en fold4_room1_mix035.wav (aprox 3 hours)
Best cost:  0.44999999999999996
Best parameters:  {'diff_th': 0.1, 'K_th': 25, 'min_event_length': 13, 'V_azi': 0.5, 'in_sd': 5, 'in_sdn': 25, 'init_birth': 0.6, 'in_cp': 0.4, 'num_particles': 30, 'event_similarity_th': 0.2}

resultados en los 10 primeros audios [fold4_room1_mix031.wav, fold4_room1_mix035.wav, fold4_room1_mix006.wav, fold4_room1_mix002.wav, fold4_room2_mix002.wav, fold1_room2_mix026.wav, fold3_room1_mix043.wav, fold1_room1_mix036.wav, fold3_room2_mix043.wav, fold3_room1_mix018.wav]
(stopped after 16 hours, estimated runtime 30 hours)
New optimized cost  0.555171035860691
New optimized parameters  {'diff_th': 0.5, 'K_th': 19, 'min_event_length': 13, 'V_azi': 5, 'in_sd': 25, 'in_sdn': 15, 'init_birth': 0.4, 'in_cp': 0.6, 'num_particles': 50, 'event_similarity_th': 0.2}


"""

import math
import random
import sys
import os
import time
import config as conf
import datetime
import numpy as np
import json
from localization_detection import localize_detect, get_groundtruth, get_evaluation_metrics


##################################################################
# Parameters

param_values = [
    [0.1, 0.3, 0.5, 0.7, 0.9],  # diff_th
    [1, 7, 13, 19, 25],  # K_th
    [1, 7, 13, 19, 25],  # min_event_length
    [0.1, 0.5, 1, 5, 10],  # V_azi
    [0, 5, 10, 15, 20, 25, 30],  # in_sd
    [0, 5, 10, 15, 20, 25, 30],  # in_sdn
    [0.2, 0.4, 0.6, 0.8],  # init_birth
    [0.2, 0.4, 0.6, 0.8],  # in_cp
    [10, 30, 50, 100],  # num_particles
    [0.2, 0.4, 0.6, 0.8],  # event_similarity_th
]
param_values_lengths = [len(p) for p in param_values]

audio_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(conf.data_folder_path) for f in fn]
json_output_folder_path = '/home/pans/datasets/DCASE2021/generated/exp2'
write_json_file = True
if write_json_file:
    create_folder(json_output_folder_path)



##################################################
# Optimization parameters
initial_temp = 1.0  # initial temperature should be calibrated
final_temp = 0.01 * initial_temp  # good practice: % of initial temperature: 0.01.
lmarkov = 20  # number of iterations on each temperature step
r = 0.90  # geometric factor for temperature decrease. It should be calibrated
nvar = 3
neighbor_dist = 1
current_temp = initial_temp
num_iters = int(lmarkov * (math.log(final_temp / initial_temp, r)))
num_max_markov = 1 # max number of markov chains allowed without cost improvement
simulate_cost = False




##################################################################
# Methods for execution of the parametric filter

def loss_function(frame_recall):
    return 1-frame_recall

def run_all_dataset(audio_files, parameters, write=False):
    # Initialize as nan, so we can track if a problem happened at any file
    num_audio_files = len(audio_files)
    EP = np.empty(num_audio_files) * np.nan
    ER = np.empty(num_audio_files) * np.nan
    DOA = np.empty(num_audio_files) * np.nan
    FR = np.empty(num_audio_files) * np.nan

    for af_idx, audio_file_name in enumerate(audio_files):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
        print("{}: {}, {}".format(af_idx, st, audio_file_name))
        # Compute the stuff
        est_event_list = localize_detect(parameters, audio_file_name)
        if len(est_event_list) == 0:
            print('Empty list. Continue')
            continue
        gt_event_list = get_groundtruth(audio_file_name)
        EP[af_idx], ER[af_idx], DOA[af_idx], FR[af_idx] = \
            get_evaluation_metrics(est_event_list, gt_event_list, parameters)

    # Average on the entire dataset and compute cost of the specific parameter combination
    meanEP = np.mean(EP[~np.isnan(EP)])
    meanER = np.mean(ER[~np.isnan(ER)])
    meanDOA = np.mean(DOA[~np.isnan(DOA)])
    meanFR = np.mean(FR[~np.isnan(FR)])
    cost = loss_function(meanFR)

    # Write to file
    if write:
        json_file_name = str(time.time()).split('.')[0] + '.json'  # UTC in seconds
        json_output_file_path = os.path.join(json_output_folder_path, json_file_name)
        write_json(json_output_file_path, parameters, meanEP, meanER, meanDOA, meanFR, cost)

    return cost, meanEP, meanER, meanDOA, meanFR

def write_json(output_file_path, parameters, mean_EP, mean_ER, mean_DOA, mean_FR, cost):
    data = {}
    data['parameters'] = []
    data['parameters'].append(parameters)
    data['result'] = []
    data['result'].append({
        'mean_EP': mean_EP,
        'mean_ER': mean_ER,
        'mean_DOA': mean_DOA,
        'mean_FR': mean_FR,
        'cost': cost,
    })
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile)




##################################################################
# Optimization method helpers

def get_random_start(param_values_lengths):
    random_indices = np.empty(len(param_values_lengths), dtype=int)
    for p_idx in range(len(param_values_lengths)):
        random_indices[p_idx] = random.randint(0, param_values_lengths[p_idx]-1)
    return random_indices

def get_neighbors(param_indices, param_values_lengths, nvar, neighbor_dist):
    new_param_indices = param_indices
    # set which params will move
    move_params_indices = random.sample(range(len(param_values_lengths)), nvar) # nvar movements are randomly defined
    for m in move_params_indices:
        # new_idx = param_indices[m] + random.randint(-1,1) # just 1 step in any direction is allowed. This could be also be a parameter
        # implemented as either +neighbor_dist or -neighbor_dist
        new_idx = param_indices[m] + (((random.randint(0,1) * 2) -1) * neighbor_dist)
        new_param_indices[m] = int(max(0, min(param_values_lengths[m]-1, new_idx)))
    return new_param_indices

def assign_parameters(param_values, param_indices):
    parameters = {}
    parameters['diff_th']               = param_values[0][param_indices[0]]
    parameters['K_th']                  = param_values[1][param_indices[1]]
    parameters['min_event_length']      = param_values[2][param_indices[2]]
    parameters['V_azi']                 = param_values[3][param_indices[3]]
    parameters['in_sd']                 = param_values[4][param_indices[4]]
    parameters['in_sdn']                = param_values[5][param_indices[5]]
    parameters['init_birth']            = param_values[6][param_indices[6]]
    parameters['in_cp']                 = param_values[7][param_indices[7]]
    parameters['num_particles']         = param_values[8][param_indices[8]]
    parameters['event_similarity_th']   = param_values[9][param_indices[9]]
    return parameters



##################################################################
# Main loop
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'short':
        # audio_files = conf.short_audio_file_list
        audio_files = [audio_files[1]]

    ##################################################
    # Run initial experiment with random start
    current_param_indices = get_random_start(param_values_lengths)
    current_parameters = assign_parameters(param_values, current_param_indices)
    current_cost = None
    if simulate_cost:
        current_cost = random.random()
    else:
        current_cost = run_all_dataset(audio_files, current_parameters, write=write_json_file)[0]
    best_cost = current_cost
    best_parameters = current_parameters

    ##################################################
    # Main loop
    iter = 0
    while current_temp > final_temp:
        print('                                                                           ')
        print('======================= ITERATION ' + str(iter+1) + ' of ' + str(num_iters) + ' =======================')
        print('== Temperature: ' + str(current_temp))
        print('current_cost', current_cost)
        print('current_parameters', current_parameters)

        ncmsm = 0 # number of markov chains without improvements
        for i in range(lmarkov):

            ##################################################
            # Compute next iteration
            new_param_indices = get_neighbors(current_param_indices, param_values_lengths, nvar, neighbor_dist)
            new_parameters = assign_parameters(param_values, new_param_indices)
            if simulate_cost:
                new_cost = random.random()
            else:
                new_cost = run_all_dataset(audio_files, current_parameters, write=write_json_file)[0]

            ##################################################
            # Assert improvement (smaller cost)
            cost_diff = current_cost - new_cost
            #print('current_cost', current_cost)
            #print('new_cost', new_cost)
            #print('cost_diff', cost_diff)

            ##################################################
            # Continue path if new parameter set improved
            if cost_diff > 0:
                print('update', math.exp(-cost_diff / current_temp))
                current_param_indices = new_param_indices
                current_parameters = new_parameters
                current_cost = new_cost
                # if current cost improves best cost, update it
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_parameters = current_parameters
                    ncmsm = 0
                    print('New optimized cost ', best_cost)
                    print('New optimized parameters ', best_parameters)
            else:
                ##################################################
                # Continue path with random probability, if new parameter set didn't improve
                rand = random.uniform(0, 1)
                if rand < math.exp(-cost_diff / current_temp):
                    print('random update', rand, math.exp(-cost_diff / current_temp))
                    current_param_indices = new_param_indices
                    current_parameters = new_parameters
                    current_cost = new_cost
                ncmsm += 1
            iter += 1
            if ncmsm > num_max_markov:
                print('Not improvements for temp=', current_temp)
                break
        current_temp = current_temp*r
    print('Finished')
    print('Best cost: ', best_cost)
    print('Best parameters: ', best_parameters)
