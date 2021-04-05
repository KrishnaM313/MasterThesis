## 
Train locally


run this file in the repo root directory to use `$(pwd)` function in export command

## Neural Network 
```bash
export REPODIR="$(pwd)"
python "$REPODIR/scripts/08_cnn_train.py" --data-path "$REPODIR/data/embeddings" --pretrained-model "$REPODIR/data/models/bert-base-uncased" --category "climate"
```

## Random Forest
```bash
export REPODIR="$(pwd)"
python "$REPODIR/scripts/09_baseline_random_forest.py" --data-path "$REPODIR/data/json_enriched" --category "climate"
```
