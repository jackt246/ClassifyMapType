#!/bin/bash

read -p "Enter path for text file which lists maps to be copied: " FILE
echo $FILE
INPUT=$(cat $FILE)
echo $INPUT

foreach file ($INPUT)
echo $file