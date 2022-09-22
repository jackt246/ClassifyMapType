#!/bin/bash

read -p "Enter path for text file which lists maps to be copied: " FILE
read - p "Enter path to folder you wish to copy these files to: " FolderName

INPUT=$(cat $FILE)


for file in $INPUT
do
cp /nfs/production/gerard/emdb/archive/staging/structures/$file/map/${file//EMD-/emd_}.map.gz $FolderName
done