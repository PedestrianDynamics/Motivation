#!/bin/bash

# Run json to xml parser
python src/json2xml.py files/bottleneck.json
if [[ $? -ne 0 ]]; then
    echo "Command failed: python src/json2xml.py files/bottleneck.json"
    exit 1
fi

# Run inifile parser
python src/inifile_parser.py files/bottleneck.json files/schema.json
if [[ $? -ne 0 ]]; then
    echo "Command failed: python src/inifile_parser.py files/bottleneck2.json files/schema.json"
    exit 1
fi

echo "All commands ran successfully."
exit 0
