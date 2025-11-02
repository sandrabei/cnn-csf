mkdir -p logs
for l in val test; do
  for exp in $(cat exp.list); do
    python3 eval.py $exp --data_list ${l}.list >logs/${l}_${exp}.txt
  done
done
grep 'IoU' logs/*
