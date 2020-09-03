#!/usr/bin/env python3

import os
import re
import sys
import jsonpickle
from os.path import join

BASE_PATH = "trained_models"
RESULT_BASE_PATH = "results"
TEST_DATA_PATH = "input/evaluation_data_UR__K"
GAN_EXP_PATTERN = re.compile("([a-zA-Z])+_gan.*")

if len(sys.argv) > 1:
	EXP_PATTERN = re.compile(sys.argv[1])
else:
	EXP_PATTERN = None

def check_disaggregate(experiment_name, resolution, appliance, experiment_id, EXPERIMENT_PATH):
	try:
	    with open(join(EXPERIMENT_PATH, 'run.json'), 'r') as file:
	        run = jsonpickle.decode(file.read())
	except:
		return

	if run['status'] != 'COMPLETED':
		return

	RESULT_FILE = join(RESULT_BASE_PATH, experiment_name, resolution, experiment_id, "{}.json".format(appliance))
	if os.path.exists(RESULT_FILE):
		return
	else:
		if EXP_PATTERN is not None:
			if EXP_PATTERN.match(experiment_name) is None:
				return
		if GAN_EXP_PATTERN.match(experiment_name) is not None:
			disagg_command = "python3 gan_g_verifier.py {exp} {appl} --id {id} --modeldir '{exp_base}/{exp}/{res}' --data '{testdir}' --resultdir '{res_base}/{exp}/{res}' --nosequences".format(
				exp_base=BASE_PATH, exp=experiment_name, appl=appliance, res=resolution, testdir=TEST_DATA_PATH, res_base=RESULT_BASE_PATH, id=experiment_id)
			print("would run '{}'...".format(disagg_command))
			return
		os.makedirs(join(RESULT_BASE_PATH, experiment_name, resolution, experiment_id), exist_ok=True)
		disagg_command = "python3 disaggregator.py {exp} {appl} --id {id} --modeldir '{exp_base}/{exp}/{res}' --data '{testdir}' --resultdir '{res_base}/{exp}/{res}' --nosequences".format(
			exp_base=BASE_PATH, exp=experiment_name, appl=appliance, res=resolution, testdir=TEST_DATA_PATH, res_base=RESULT_BASE_PATH, id=experiment_id)
		print("running '{}'...".format(disagg_command))
		#print("would run '{}'...".format(disagg_command))
		#return
		if os.system(disagg_command) != 0:
			sys.exit(1)


# traverse trained models directory
for experiment_name in sorted(os.listdir(BASE_PATH)):
	EXPERIMENT_BASE_PATH = join(BASE_PATH, experiment_name)
	if not os.path.isdir(EXPERIMENT_BASE_PATH):
		continue
	for resolution in sorted(os.listdir(EXPERIMENT_BASE_PATH)):
		RESOLUTION_BASE_PATH = join(EXPERIMENT_BASE_PATH, resolution)
		if not os.path.isdir(RESOLUTION_BASE_PATH):
			continue
		for appliance in sorted(os.listdir(RESOLUTION_BASE_PATH)):
			APPLIANCE_BASE_PATH = join(RESOLUTION_BASE_PATH, appliance)
			if not os.path.isdir(APPLIANCE_BASE_PATH):
				continue
			for experiment_id in sorted(os.listdir(APPLIANCE_BASE_PATH)):
				EXPERIMENT_PATH = join(APPLIANCE_BASE_PATH, experiment_id)
				if not os.path.isdir(EXPERIMENT_PATH):
					continue
				check_disaggregate(experiment_name, resolution, appliance, experiment_id, EXPERIMENT_PATH)
