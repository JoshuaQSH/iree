#!/usr/bin/env bash

DATE=`date "+%Y%m%d"`
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`

# run the tf and save the log
python run.py > ./logs/log_${DATE_WITH_TIME}.log
