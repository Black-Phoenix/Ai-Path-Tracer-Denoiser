#!/bin/bash

target="scenes_created/$1"
train="Training_data/$1"
let count=0
for f in "$target"/*
do
	A=$target/$(basename $f)
    if [ ! -d $train/${A:16:-4}/ ]; then
	  mkdir -p $train/${A:16:-4}/;
	fi
	#echo $target$(basename $f)
    ./cis565_path_tracer.exe $target$(basename $f)
    #break
done

