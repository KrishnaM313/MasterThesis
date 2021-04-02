import os
from tools_data import getBaseDir
from transformers import BertForSequenceClassification

modelName = "bert-base-uncased"

repoDir = getBaseDir()


model = BertForSequenceClassification.from_pretrained(
                                    modelName,
                                    num_labels = 9,
                                    output_attentions = False,
                                    output_hidden_states = False)
localPretrainedModelPath = os.path.join(repoDir, "data", "models", modelName)
model.save_pretrained(localPretrainedModelPath)
# ds.upload(
#     src_dir=localPretrainedModelPath,
#     target_path=modelName,
#     overwrite=False,
# )