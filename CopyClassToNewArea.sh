#!/bin/bash

#read -p "Enter path for text file which lists maps to be copied: " FILE
#read -p "Enter path to folder you wish to copy these files to: " FolderName

FILE=Analysis/IPETList.txt
OldFolder=/hps/nobackup/gerard/emdb/ClassifyMapType/Classes/Tomograms
NewFolder=/hps/nobackup/gerard/emdb/ClassifyMapType/Classes/IPET

INPUT=$(cat $FILE)


for file in $INPUT
do
echo $OldFolder/*$file* $NewFolder
mv $OldFolder/*$file* $NewFolder
done