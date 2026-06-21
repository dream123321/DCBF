#!/bin/bash

module purge
source /share/home/xill/hj/hj_app/dcbf_continue/dcbf_one-button_deployment/activate.sh
dcbf run dcbf.init_dataset.vasp.qiming.json
