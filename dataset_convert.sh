#!/bin/bash

source /srv/envs/kai/miniconda3/bin/activate
cd /srv/experiments/neuralnilm-pytorch

EXP=dataset_final_URED_
#EXP=dataset_final__R___

IFS=";"

APPLIANCES=(\
	"dish washer" \
	"washing machine"\
	"tumble dryer"\
	"kettle" \
	"microwave" \
	"television" \
	"fridge" \
	)

for APPLIANCE in ${APPLIANCES[@]}; do
python ${EXP}.py dataset.convert with dataset.TARGET_APPLIANCE="${APPLIANCE}" $@
done
