#!/bin/bash
for dim_1 in 10 20 30 40 50
do
	for dim_3 in 32 64
	do
		for op in 'cp' 'tucker'
		do
			python cp_tucker.py --time 1 $op \
			--origin $dim_1 $dim_1 $dim_3 \
			--core 5 5 16
		done
	done
done
