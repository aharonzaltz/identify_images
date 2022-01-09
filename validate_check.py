from data_loader_config import training_dataloader

total = 0
for index, (x, y) in enumerate(training_dataloader):
    print (x.shape)
    if not x.shape[1] == 4761:
        total+=1
        # del training_dataloader[index]
        # print ("before delete", x.shape, y.shape)
print (total)
