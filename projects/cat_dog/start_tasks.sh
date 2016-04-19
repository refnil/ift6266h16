#!/bin/bash

number_of_task=$(python config.py)

for i in `seq 1 $number_of_task`;
do
    task_id=$(qsub task_multi.pbs)
    echo $i > .$task_id.id
    sed "s/TOKEN/$task_id/" run.pbs | qsub
done
