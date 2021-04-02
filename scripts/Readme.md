## 
Train locally


run this file in the repo root directory to use `$(pwd)` function in export command

```bash
export REPODIR="$(pwd)"
python "$REPODIR/scripts/08_cnn_train.py" --data-path "$REPODIR/data/embeddings" --pretrained-model "$REPODIR/data/models/bert-base-uncased" --category "climate"
```