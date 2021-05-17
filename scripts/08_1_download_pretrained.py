import os

from transformers import BertForSequenceClassification

from tools_data import getBaseDir

modelName = "bert-base-uncased"

repoDir = getBaseDir()


model = BertForSequenceClassification.from_pretrained(
    modelName,
    num_labels=9,
    output_attentions=False,
    output_hidden_states=False)
localPretrainedModelPath = os.path.join(repoDir, "data", "models", modelName)
model.save_pretrained(localPretrainedModelPath)
