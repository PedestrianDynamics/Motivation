#!/bin/bash
# usage:
#./run.sh
#jps-model: demo_02.py
#inifile: bottleneck.json
#trajectory: 02.txt
#flag: 1

EXPECTED_ARGS=4
jpsvis=~/workspace/jupedsim/jpscore/build/bin/jpsvis.app/Contents/MacOS/jpsvis
CONVERTER=~/workspace/jupedsim/jpscore/scripts/petrack2jpsvis.py
if [ "$#" -eq "$EXPECTED_ARGS" ]
then
    echo "INFO : Running $0 with $1 $2"
    PYTHONPATH=~/workspace/jupedsim/jpscore/build/lib:~/workspace/jupedsim/jpscore/python_modules/jupedsim/ python $1 $2
    if [ $? -eq 0 ]; then
        if [ $4 -eq 1 ]
        then
            echo "INFO : Run petrack2jpsvis converter"
            $CONVERTER $3
            echo "INFO : run jpsvis"
            $jpsvis jps_$3
        fi
    else
        printf "\n -----\nSimulation errors!\n"
    fi
else
    printf "Invalid number of arguments.\nUsage %s python-script 0|1\n" $0
fi

     
