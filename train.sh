#!/bin/bash
function pause(){
   read -p "$*"
}
target="scenes/scenes_created_active/$1"
train="Training_data/$1"
let count=0
for f in "$target"/*
do
	scene_name=$(basename $f)
	scene_name=${scene_name%%.*}
	echo $scene_name
    if [ ! -d $train/"GroundTruth"/ ]; then
	  mkdir -p $train/"GroundTruth"/;
	fi
    if [ ! -d $train/"Albedos"/ ]; then
	  mkdir -p $train/"Albedos"/;
	fi
	if [ ! -d $train/"RGB"/ ]; then
	  mkdir -p $train/"RGB"/;
	fi
	if [ ! -d $train/"Normals"/ ]; then
	  mkdir -p $train/"Normals"/;
	fi
	if [ ! -d $train/"Depth"/ ]; then
	  mkdir -p $train/"Depth"/;
	fi
	#echo $target$(basename $f)
    ./build/bin/Release/cis565_path_tracer.exe $target/$(basename $f)
    break
done

pause "Press key to end"