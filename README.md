# mgr

`pip install -r requirements.txt --quiet`


config.ini:
```ini
[cometml]
apikey = ...
projectname = mgr
workspace = plutasnyy
```

prepare zip with data:
`find data -name "*data.csv" | xargs zip data.zip`

Example run:
```bash
sbatch --job-name '100proto' -w xeon-09 --gres=gpu:01 -c 8 --wrap "python3 src/train.py \
    --run-name 'log sim' \
    --project-name 'mgr-bench' \
    --datasets 'hotel' \
    --model 'protoconv' -lr 1e-3 \
    --seed 0 \
    --epoch 30 \
    --batch-size 32 \
    --pc-conv-filters 64 \
    --pc-conv-filter-size 3 \
    --pc-conv-stride 1 \
    --pc-conv-padding 1 \
    --pc-project-prototypes-every-n 2 \
    --pc-prototypes-init 'rand' \
    --pc-number-of-prototypes -1 \
    --pc-sim-func 'log' \
    --pc-ce-loss-weight 0.99 \
    --pc-cls-loss-weight 0.005 \
    --pc-sep-loss-weight 0.005 \
    --pc-separation-threshold 1 \
    --pc-l1-loss-weight 1e-2 \
    --pc-visualize False \
    --fold 5"
    #--fold 1 -fdr 1 --no-logger"` For test run
    ``