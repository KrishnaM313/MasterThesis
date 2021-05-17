from sklearn.metrics import multilabel_confusion_matrix
import json
from icecream import ic


def prettyPrint(json_string: str):
    print(json.dumps(json_string, indent=4))


def logValue(run, name, value, verbose=False):
    logValues(run, {name: value}, verbose=verbose)


def logValues(run, result: dict, verbose=False):
    for key in result:
        if verbose:
            print("{} : {}".format(key, str(result[key])))
        run.log(key, result[key])


def logConfusionMatrix(run, labels_epoch, predicted_epoch, labels=None, labelsRange=9, verbose=False):
    if verbose:
        print(labels_epoch)
        print(predicted_epoch)

    if labels is None:
        labels = [int(x) for x in range(labelsRange)]

    if verbose:
        print(labels)

    confusion_matrices = multilabel_confusion_matrix(
        labels_epoch, predicted_epoch, labels=labels)

    for i, label in enumerate(labels):
        if verbose:
            print("Label #%s %s" % (i, label))
        confusion_matrix = confusion_matrices[i]
        logJSON = {
            "schema_type": "confusion_matrix",
            "schema_version": "1.0.0",
            "data": {
                "class_labels": labels,
                "matrix": [[int(y) for y in x] for x in confusion_matrix]
            }
        }
        print(logJSON)
        run.log_confusion_matrix('confusion_matrix_' + label, logJSON)


def printModelParameters(model):
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(
        len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
