
model='gru'

mkdir results 2>/dev/null
mkdir results/${model} 2>/dev/null

echo ${model}...

python run.py --model ${model} --epochs 30