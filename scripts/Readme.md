## 
Train locally


run this file in the repo root directory to use `$(pwd)` function in export command

## Neural Network 
```bash
export REPODIR="$(pwd)"
python "$REPODIR/scripts/08_cnn_train.py" --data-path "$REPODIR/data/embeddings" --pretrained-model "$REPODIR/data/models/bert-base-uncased" --category "climate" --batch-size 1 --demo-limit 1 
```

## Random Forest
```bash
export REPODIR="$(pwd)"
python "$REPODIR/scripts/09_baseline_random_forest.py" --data-path "$REPODIR/data/embeddings" --category "climate" --labels "partyGroupIdeology"
```
alternative --labels "leftRightPosition"
