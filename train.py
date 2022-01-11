
import matplotlib

from data_loader_config import training_dataloader, valid_dataloader
from test_items import loss_func, compute_accuracy_and_loss

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm

from config import net_file_name, path_to_training_data, path_to_validation_data, batch_size, training_cycles
from model import Net
import torch
import torch.optim as optim

print ("Start")

net = Net()

training_loss_vs_cycle = []
validation_loss_vs_cycle = []

training_acc_vs_cycle = []
validation_acc_vs_cycle = []

pbar = tqdm(range(training_cycles))
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in pbar:

    if len(validation_loss_vs_cycle) > 1:
        pbar.set_description('val acc:' + '{0:.5f}'.format(validation_acc_vs_cycle[-1]) +
                             ', train acc:' + '{0:.5f}'.format(training_acc_vs_cycle[-1]))

    net.train()  # put the net into "training mode"
    for input, correct_answer in training_dataloader:

        if torch.cuda.is_available():
            input = input.cuda()
            correct_answer = correct_answer.cuda()

            # add the basic training loop here

        optimizer.zero_grad()

        output = net(input)
        loss = loss_func(output, correct_answer)

        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode

    train_acc, train_loss = compute_accuracy_and_loss(training_dataloader, net)
    valid_acc, valid_loss = compute_accuracy_and_loss(valid_dataloader, net)

    training_loss_vs_cycle.append(train_loss)
    training_acc_vs_cycle.append(train_acc)

    validation_acc_vs_cycle.append(valid_acc)
    validation_loss_vs_cycle.append(valid_loss)

    # save the model if the validation loss has decreased
    if len(validation_loss_vs_cycle) == 1 or validation_loss_vs_cycle[-2] > validation_loss_vs_cycle[-1]:
        torch.save(net.state_dict(), net_file_name + '.pt')

fig, ax = plt.subplots(1, 2, figsize=(8, 3))

ax[0].plot(training_loss_vs_cycle, label='training')
ax[0].plot(validation_loss_vs_cycle, label='validation')

ax[1].plot(training_acc_vs_cycle)
ax[1].plot(validation_acc_vs_cycle)

plt.show()
