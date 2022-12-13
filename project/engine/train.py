import torch
import torch.optim as optim

def train(model, device, train_loader, optimizer, criterion):
    model.train()

    # accuracy stats collection
    correct = 0
    processed = 0
    epoch_loss = 0

    for batch_id, batch in enumerate(train_loader):
        image = batch['image'].to(device)
        question = batch['question'].to(device)
        target = batch['answer'].to(device)

        optimizer.zero_grad()

        pred = model(image, question)                      
        loss = criterion(pred, target)

        # calculate gradients
        loss.backward()
        # optimizer
        optimizer.step()

        # metrics
        predicted_digits = pred.argmax(dim=1, keepdim=True)
        correct += predicted_digits.eq(target.view_as(predicted_digits)).sum().item()
        processed += len(image)
        epoch_loss += loss.item()
    
    epoch_accuracy = (100 * correct) / processed
    epoch_loss /= len(train_loader)
    print(f"Training-----Epoch Accuracy {round(epoch_accuracy, 2)}\tEpoch Loss {round(epoch_loss, 4)}")
    return epoch_accuracy, epoch_loss