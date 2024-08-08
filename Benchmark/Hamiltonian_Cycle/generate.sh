#!/bin/bash

num_of_vertices=(10 20 30 40 50)

for size in "${num_of_vertices[@]}"; do
    python3 ./gen_hybrid.py --size=$size
    python3 ./hybrid2wcnf.py --size=$size
done