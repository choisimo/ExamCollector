#!/usr/bin/env bash

echo "" > ./all_code.txt

find /c/WorkSpace/git/ExamCollector/models/custom_yolo/ -type f ! -name '*.pt' ! -name '*.sqlite' ! -name '*.sql' ! -name '*.log' ! -name '*.pyc' ! -name '*.jpg' ! -name '*.png' ! -name '*.txt' ! -name '*.cache' -exec sh -c '
    echo "==============================="
    echo "FILE: $1"
    echo "==============================="
    cat "$1"
    echo
  ' _ {} \;   > ./all_code.txt
