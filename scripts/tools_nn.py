from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import numpy as np
import torch
from tqdm import tqdm
from icecream import ic

def evaluateResult(labels_list: torch.Tensor,predicted_list: torch.Tensor,epoch: int=None, prefix=""):
    # calculating the classification report with sklearn    
    # classification_report_json = classification_report(labels, predicted, output_dict=True, zero_division=0)
    
    predicted = torch.flatten(predicted_list)
    labels = torch.flatten(labels_list)


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
            prefix+"mean_absolute_error" : mean_absolute_error(labels, predicted),
            prefix+"mean_squared_error": mean_squared_error(labels, predicted),
            prefix+"sqrt_mean_squared_error": np.sqrt(mean_squared_error(labels, predicted)),
            prefix+"accuracy_score": accuracy_score(labels, predicted),
            prefix+"f1_score": f1_score(labels, predicted, average="micro")
        }
    if epoch is not None:
        result["epoch"] = epoch
    return result

def evaluateModel(model, testDataloader, device, demoLimit=0, verbose=False, prefix=""):
    if verbose:
        print("##### testing #####")
    
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    test_labels = []
    test_predictions = []

    for i, batch in enumerate(tqdm(testDataloader)):
           # print(ic(torch.sum(batch["input_ids"])))

            if (demoLimit>0) and (i>demoLimit):
                break

            #print(ic(batch))
            labels = batch["labels"].to(device)
            
            inputs = {
                'input_ids':        batch["input_ids"].to(device),
                'attention_mask':   batch["attention_mask"].to(device),
                'token_type_ids':   batch["token_type_ids"].to(device)
            }

            #batch = batch.to(device)
            #batch = tuple(key.to(device) for key in batch)

            with torch.no_grad():
                outputs = model(**inputs)
                #output = np.asarray(ic(outputs_tuple))           

            test_batch_predicted = torch.argmax(outputs[1], dim=1)
            #print(ic(test_batch_predicted))
            
            test_labels.append(torch.flatten(labels.cpu()))
            test_predictions.append(torch.flatten(test_batch_predicted.cpu()))


    test_labels = torch.cat(test_labels)
    test_predictions = torch.cat(test_predictions)

    ic(test_labels)
    ic(test_predictions)

    test_result = evaluateResult(test_labels,test_predictions, prefix=prefix)
    if verbose:
        print(test_result)
    return test_result