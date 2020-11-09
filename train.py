from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter()  # tensorboard writer


def adjust_learning_rate(optimizer, epoch, lr_initial):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = lr_initial * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, total_epoch, trainLoader, validLoader, optimizer, lr_initial, criterion, device):
    iter = 0  # training iteration
    iter_valid = 0  # validation iteration
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        valid_total = 0  # total validation number
        valid_correct = 0  # correct prediction number
        adjust_learning_rate(optimizer, epoch, lr_initial)
        running_loss = 0.0
        print('running epoch: {}, lr: {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        model.train()
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # batch loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            iter += 1
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

            # Record training loss from each epoch into the writer
            writer.add_scalar('Train/Loss', loss.item(), iter)
            writer.flush()

        # validation
        model.eval()
        valid_loss = 0.0
        for i, data in enumerate(validLoader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            iter_valid += 1
            valid_loss += loss.item()
            if i % 40 == 39:  # print every 40 mini-batches
                print('[%d, %5d] valid_loss: %.3f' %
                      (epoch + 1, i + 1, valid_loss / 40))
                valid_loss = 0.0

            # validation accuracy
            _, predicted = torch.max(outputs, 1)
            valid_total += predicted.nelement()
            valid_correct += torch.sum(predicted == labels).item()

            # Record training loss from each epoch into the writer
            writer.add_scalar('Valid/Loss', loss.item(), iter_valid)
            writer.flush()
        valid_acc = valid_correct / valid_total
        print('valid_acc: {}', valid_acc)
        writer.add_scalar('Valid/Accuracy', valid_acc, epoch + 1)
        writer.flush()

    writer.close()
    print('Finished Training')
