#!/bin/bash

#=======================
jpsvis=~/workspace/jupedsim/jpscore/build/bin/jpsvis.app/Contents/MacOS/jpsvis
CONVERTER=~/workspace/jupedsim/jpscore/scripts/petrack2jpsvis.py
python_path=~/workspace/jupedsim/jpscore/build/lib:~/workspace/jupedsim/jpscore/python_modules/jupedsim/
#=======================

EXPECTED_ARGS=4

if [[ "$#" -ne "$EXPECTED_ARGS" ]]; then
    echo "Invalid number of arguments."
    echo "Usage: $0 <python-script> <input-file> <output-file> <run-jpsvis>"
    exit 1
fi

PYTHON_SCRIPT="$1"
INPUT_FILE="$2"
OUTPUT_FILE="$3"
RUN_JPSVIS="$4"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script file not found: $PYTHON_SCRIPT"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi


echo "INFO: Running $PYTHON_SCRIPT with $INPUT_FILE $OUTPUT_FILE"

PYTHONPATH=$python_path python "$PYTHON_SCRIPT" "$INPUT_FILE" "$OUTPUT_FILE"

if [[ $? -ne 0 ]]; then
    echo "ERROR: Simulation errors!"
    exit 1
fi

if [[ "$RUN_JPSVIS" -eq 1 ]]; then
    echo "INFO: Running petrack2jpsvis converter"
    # create jps_output_file
    $CONVERTER "$OUTPUT_FILE"
    # create geometry.xml file
    echo "INFO: Running json2xml converter"
    python src/json2xml.py "$INPUT_FILE"
    echo "INFO : run jpsvis"
    $jpsvis jps_$OUTPUT_FILE

fi

     
