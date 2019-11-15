#!/bin/bash
function pause(){
   read -p "$*"
}
target="scenes/auto_gen_scenes/$1"
train="Training_data/$1"
let count=0
for f in "$target"/*
do
	scene_name=$(basename $f)
	scene_name=${scene_name%%.*}
	echo $scene_name
	A=$target/$scene_name
    if [ ! -d $train/$scene_name/ ]; then
	  mkdir -p $train/$scene_name/;
	fi
	#echo $target$(basename $f)
    ./cis565_path_tracer.exe $target/$(basename $f)
    break
done

pause "Press key to end"