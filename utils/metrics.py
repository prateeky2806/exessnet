import numpy as np

def get_forgetting_metric(acc_arr, bwt=False, return_mean=False):
    num_tasks = acc_arr.shape[0]
    max_accs = np.max(acc_arr[:-1,:-1], axis=0)
    last_accs = acc_arr[-1, :-1]
    if bwt:
        task_forgetting = last_accs - max_accs
    else:
        task_forgetting = max_accs - last_accs
    if return_mean:
        return np.array(task_forgetting).mean()
    return np.array(task_forgetting)

def get_forward_transfer(acc_arr, random_accs, return_mean=False):
    fwt = []
    for i in range(1, len(acc_arr)):
        fwt.append(acc_arr[i-1, i] - random_accs[i])
    if return_mean:
        return np.array(fwt).mean()
    return np.array(fwt)