#!/bin/bash

rsync -av --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/meizhiyu.mzy/verl /storage/openpsi/users/meizhiyu.mzy/run_verl
chmod -R 755 /storage/openpsi/users/meizhiyu.mzy/run_verl

sbatch $1
