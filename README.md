# mgr

`pip install -r requirements.txt --quiet`

config.ini:
```ini
[cometml]
apikey = ...
projectname = mgr-bench2
workspace = plutasnyy
```

prepare zip with data:
`find data -name "*data.csv" | xargs zip data.zip`
`python -m spacy download en`

Reproduce CNN:
```bash
sbatch --job-name 'cnn' -w xeon-09 --gres=gpu:01 -p all --wrap "python3 src/train.py \
    --run-name 'cnn' \
    --project-name mgr-bench2\
    --datasets 'all' \
    --model 'cnn' \
    -lr 1e-3 \
    --seed 0 \
    --epoch 30 \
    --batch-size 32 \
    --cnn-conv-filters 64 \
    --cnn-filter-size 5 \
    --fold 5"
```

Example run:
```bash
sbatch --job-name 'ProtoCNN' -w xeon-09 --gres=gpu:01 -p all --wrap "python3 src/train.py \
    --run-name 'ProtoCNN' \
    --project-name mgr-bench2\
    --datasets 'all' \
    --model 'protoconv' \
    -lr 1e-3 \
    --seed 0 \
    --epoch 30 \
    --batch-size 32 \
    --pc-conv-filters 64 \
    --pc-conv-filter-size 5 \
    --pc-project-prototypes-every-n 1 \
    --pc-prototypes-init 'rand' \
    --pc-number-of-prototypes 16 \
    --pc-sim-func 'log' \
    --pc-ce-loss-weight 0.99 \
    --pc-cls-loss-weight 0.005 \
    --pc-sep-loss-weight 0.005 \
    --pc-separation-threshold 1 \
    --pc-l1-loss-weight 1e-2 \
    --pc-visualize True \
    --pc-dynamic-number True \
    --fold 5"
    #--fold 1 -fdr 1 --no-logger"` For test run
    ``