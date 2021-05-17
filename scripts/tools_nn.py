from typing import List, Union

import numpy as np
import torch
from icecream import ic
from sklearn.metrics import (classification_report, jaccard_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from tqdm import tqdm

from tools_logging import logValue
from tools_parties import getPartyIdeologyAssociationsList


def evaluateResult(
        labels: Union[torch.Tensor, np.ndarray],
        predicted: Union[torch.Tensor, np.ndarray],
        target_names: List[str] = None,
        epoch: int = None,
        prefix="",
        demoLimit=0):

    if torch.is_tensor(predicted):
        predicted = torch.flatten(predicted)
    if torch.is_tensor(labels):
        labels = torch.flatten(labels)

    result = {
        prefix+"accuracy": jaccard_score(labels, predicted, average="weighted"),
        prefix+"mean_absolute_error": mean_absolute_error(labels, predicted),
        prefix+"mean_squared_error": mean_squared_error(labels, predicted),
        prefix+"sqrt_mean_squared_error": np.sqrt(mean_squared_error(labels, predicted)),
        prefix+"r2_score": r2_score(labels, predicted),
    }
    print("{}_classification_report".format(prefix))
    print(classification_report(labels, predicted,
          target_names=getPartyIdeologyAssociationsList()))

    if epoch is not None:
        result["epoch"] = epoch
    return result


def evaluateModel(model, testDataloader, device, run, demoLimit=0, verbose=False, prefix=""):
    if verbose:
        print("##### testing #####")

    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    test_labels = []
    test_predictions = []

    # Tracking variables
    total_accuracy = 0
    total_loss = 0
    nb_eval_steps = 0

    for i, batch in enumerate(tqdm(testDataloader)):
        if (demoLimit > 0) and (i > demoLimit):
            break

        batch_input_ids = batch["input_ids"].to(device)
        batch_token_type_ids = batch["token_type_ids"].to(device)
        batch_attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        with torch.no_grad():
            (loss, logits) = model(
                input_ids=batch_input_ids,
                token_type_ids=batch_token_type_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )

        total_loss += loss.item()

        logits = logits.detach().cpu()
        batch_labels = batch_labels.detach().cpu()

        test_batch_predicted = torch.argmax(logits, 1)
        test_labels.append(torch.flatten(batch["labels"]))
        test_predictions.append(test_batch_predicted.cpu())

    avg_loss = total_loss / len(testDataloader)
    logValue(run, prefix+"avg_loss", avg_loss)

    test_labels = torch.cat(test_labels)
    test_predictions = torch.cat(test_predictions)

    test_result = evaluateResult(test_labels, test_predictions, prefix=prefix)
    if verbose:
        print(test_result)
    return test_result
