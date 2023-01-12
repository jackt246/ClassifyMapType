#!/bin/bash

FILE=IPETList.txt

INPUT=$(cat $FILE)


for file in $INPUT
do
mv /hps/nobackup/gerard/emdb/TomogramCheck/Tomograms/$file/${file}.map.gz /hps/nobackup/gerard/emdb/TomogramCheck/NonTomograms/
echo ${file}
done