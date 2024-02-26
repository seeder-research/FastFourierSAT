echo "weighted maxcut"
python xor.py ../benchmark/examples/maxcut/small.wcnfp --weighted 1
echo "parity learning"
python xor.py ../benchmark/examples/parity/small.cnfp --tolerance 8
echo "cardinality constraint"
python card.py ../benchmark/examples/card/small.cnfp