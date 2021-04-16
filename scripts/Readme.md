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


```bash
/Users/michael/bin/anaconda3/envs/masterthesis/bin/python /Users/michael/workspaces/MasterThesis/scripts/11_cnn_val.py --models-path '/Users/michael/Downloads/model' --labels partyGroupIdeology --category health --train-share 0.80 --test-share 0.15 --model-typ forest
```