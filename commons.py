import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
basefilename = None


def calculate_accuracy(scores, targets, k=1):
    f = torch.FloatTensor()
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    
    return (correct_total.item())/batch_size
