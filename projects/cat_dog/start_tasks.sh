#!/bin/bash

number_of_task=$(python config.py)

for i in `seq 1 $number_of_task`;
do
    task_id=$(qsub train_many.pbs)
    echo $i > .$task_id.id
done
