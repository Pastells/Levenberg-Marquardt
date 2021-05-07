#!/bin/bash
gcc="gcc -ofast -o"
${gcc} conjugate_gradient conjugate_gradient.c -lm
${gcc} levenberg_marquardt levenberg_marquardt.c -lm
