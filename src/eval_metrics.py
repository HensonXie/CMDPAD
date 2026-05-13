import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def eval_aff_rec(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    binary_truth_non0 = test_truth > 0
    binary_preds_non0 = test_preds > 0
    f_score_non0 = f1_score(
        binary_truth_non0, binary_preds_non0, average='weighted')
    acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

    return {'mae': mae, 'corr': corr, 'acc5': mult_a5, 'acc3': mult_a3, 'f1_n0': f_score_non0, 'acc2_n0': acc_2_non0}

def eval_aff_pre(results, truths):
    return eval_aff_rec(results, truths, exclude_zero=True)

def eval_per_rec(results, truths):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    mae = np.mean(np.absolute(test_preds - test_truth))
    rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    binary_truth = test_truth > 0
    binary_preds = test_preds > 0
    f_score = f1_score(binary_truth, binary_preds, average='weighted')
    acc_2 = accuracy_score(binary_truth, binary_preds)

    return {'mae': mae, 'rmse': rmse, 'corr': corr, 'f1': f_score, 'acc2': acc_2}