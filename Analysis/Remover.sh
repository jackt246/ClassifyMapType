#!/bin/bash

#read -p "Enter path for text file which lists maps to be copied: " FILE
#read -p "Enter path to folder you wish to copy these files to: " FolderName

FILE=remove.txt
FolderName=Tomograms

INPUT=$(cat $FILE)


for file in $INPUT
do
rm /hps/nobackup/gerard/emdb/TomogramCheck/Tomograms/${file}.map.gz
done