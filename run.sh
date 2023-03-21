#!/bin/bash

EXPECTED_ARGS=2
jpsvis=~/workspace/jupedsim/jpscore/build/bin/jpsvis.app/Contents/MacOS/jpsvis
CONVERTER=~/workspace/jupedsim/jpscore/scripts/petrack2jpsvis.py
if [ "$#" -eq "$EXPECTED_ARGS" ]
then
    echo "INFO : Running $0 with $1 $2"
    PYTHONPATH=~/workspace/jupedsim/jpscore/build/lib:~/workspace/jupedsim/jpscore/python_modules/jupedsim/ python $1
    if [ $? -eq 0 ]; then
        if [ $2 -eq 1 ]
        then
            echo "INFO : Run petrack2jpsvis converter"
            $CONVERTER out.txt
            echo "INFO : run jpsvis"
            $jpsvis jps_out.txt
        fi
    else
        printf "\n -----\nSimulation errors!\n"
    fi
else
    printf "Invalid number of arguments.\nUsage %s python-script 0|1\n" $0
fi

     
