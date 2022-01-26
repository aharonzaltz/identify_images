
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
        for _input, correct_answer in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                _input = _input.cuda()
                correct_answer = correct_answer.cuda()
            answer = net(_input)

            loss += loss_func(answer, correct_answer).item()

            answer = torch.argmax(answer, dim=1)
            correct += len(torch.where(answer == correct_answer)[0])
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
            answer = net(x)
            answer = torch.argmax(answer, dim=1)
    return answer
