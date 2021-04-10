from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, r2_score, classification_report, jaccard_score
import numpy as np
import torch
from tqdm import tqdm
from icecream import ic
from tools_logging import logValue
from typing import List, Union
from tools_parties import getPartyIdeologyAssociations


def evaluateResult(
        labels: Union[torch.Tensor, np.ndarray],
        predicted: Union[torch.Tensor, np.ndarray],
        target_names: List[str]=None,
        epoch: int=None, 
        prefix="", 
        demoLimit=0):
    # calculating the classification report with sklearn    
    # classification_report_json = classification_report(labels, predicted, output_dict=True, zero_division=0)
    
    if torch.is_tensor(predicted):
        predicted = torch.flatten(predicted)
    if torch.is_tensor(labels):
        labels = torch.flatten(labels)


    #predicted = np.array([predicted_list])
    #print(ic(predicted))
    #labels = np.array([labels_list])
    #print(ic(labels))



    result = {
            #"correct" : correct,
            #"total" : total,
            #"accuracy" : running_acc, 
            #"stage" : 'testing',
            #"classification_report_json" : classification_report_json,
#               "classification_report_str" : classification_report_str,
            #"predicted": predicted_str,
            #"labels": labels_epoch_str,
            prefix+"accuracy" : jaccard_score(labels,predicted,average="weighted"),
            prefix+"mean_absolute_error" : mean_absolute_error(labels, predicted),
            prefix+"mean_squared_error": mean_squared_error(labels, predicted),
            prefix+"sqrt_mean_squared_error": np.sqrt(mean_squared_error(labels, predicted)),
            prefix+"r2_score": r2_score(labels, predicted),
        }
    #if target_names is not None:
    #ic(getPartyIdeologyAssociations().keys())
    #print(classification_report(labels, predicted))
        #prefix+"f1_score": f1_score(labels, predicted, average="micro")
        #prefix+"accuracy_score": accuracy_score(labels, predicted),


    
    if epoch is not None:
        result["epoch"] = epoch
    return result

def evaluateModel(model, testDataloader, device, run, demoLimit=0, verbose=False, prefix="" ):
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
           # print(ic(torch.sum(batch["input_ids"])))

            if (demoLimit>0) and (i>demoLimit):
                break

            #print(ic(batch))
            # labels = batch["labels"].to(device)
            
            # inputs = {
            #     'input_ids':        batch["input_ids"].to(device),
            #     'attention_mask':   batch["attention_mask"].to(device),
            #     'token_type_ids':   batch["token_type_ids"].to(device)
            # }

            batch_input_ids = batch["input_ids"].to(device)
            batch_token_type_ids = batch["token_type_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)


            #batch = batch.to(device)
            #batch = tuple(key.to(device) for key in batch)

            with torch.no_grad():
                (loss, logits) = model(
                    input_ids=batch_input_ids,
                    token_type_ids=batch_token_type_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                #output = np.asarray(ic(outputs_tuple))           

            total_loss += loss.item()

            logits = logits.detach().cpu()
            batch_labels = batch_labels.detach().cpu()

            

            #ic(loss)
            #ic(logits)
            #test_batch_predicted = torch.argmax(outputs, dim=1)
            test_batch_predicted = torch.argmax(logits, 1)
            #print(ic(test_batch_predicted))
            #total_accuracy += accuracy_score(test_batch_predicted, batch_labels)
            
            test_labels.append(torch.flatten(batch["labels"]))
            test_predictions.append(test_batch_predicted.cpu())

    #avg_accuracy = total_accuracy / len(testDataloader)
    #logValue(run, prefix+"avg_accuracy",avg_accuracy)

    avg_loss = total_loss / len(testDataloader)
    logValue(run, prefix+"avg_loss",avg_loss)


    test_labels = torch.cat(test_labels)
    test_predictions = torch.cat(test_predictions)

    test_result = evaluateResult(test_labels,test_predictions, prefix=prefix)
    if verbose:
        print(test_result)
    return test_result