#!/bin/bash

nsys profile -o test_report --stats=true --trace=cuda,osrt,nvtx --sample=cpu_host -- python3 test.py
