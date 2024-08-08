#!/bin/bash

num_of_vertices=(100 200 300 400 500)

for size in "${num_of_vertices[@]}"; do
    python3 ./gen_hybrid.py --size=$size
    python3 ./hybrid2wcnf.py --size=$size
done