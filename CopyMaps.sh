#!/bin/bash

read -p "Enter path for text file which lists maps to be copied: " FILE
echo $FILE
INPUT=$(cat $FILE)
echo $INPUT

for file in $INPUT
do
echo $file
done