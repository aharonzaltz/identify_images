
import torch.nn as nn
import torch


loss_func = nn.CrossEntropyLoss()

def compute_accuracy_and_loss(dataloader, net):
    total = 0
    correct = 0
    loss = 0

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    n_batches = 0
    with torch.no_grad():
        for input, correct_answer in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                input = input.cuda()
                correct_answer = correct_answer.cuda()
            pred = net(input)

            loss += loss_func(pred, correct_answer).item()

            pred = torch.argmax(pred, dim=1)
            correct += len(torch.where(pred == correct_answer)[0])
            total += len(correct_answer)
    loss = loss / n_batches
    return float(correct) / total, loss


def compute_accuracy_item(item, net):
    answer = 0
    with torch.no_grad():
        for x, y in item:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            pred = net(x)
            pred = torch.argmax(pred, dim=1)
            answer = pred
    return answer



