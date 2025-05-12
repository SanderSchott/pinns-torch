#!/bin/bash

nsys profile --trace=cuda,osrt,nvtx --sample=process-tree --output=test_report python3 testing.py
nsys stats test_report.nsys-rep
