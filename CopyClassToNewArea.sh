#!/bin/bash

#read -p "Enter path for text file which lists maps to be copied: " FILE
#read -p "Enter path to folder you wish to copy these files to: " FolderName

FILE=Analysis/IPETList.txt
OldFolder=Classes/Tomograms
NewFolder=Classes/IPET

INPUT=$(cat $FILE)


for file in $INPUT
do
mv $OldFolder/$file* $NewFolder
done