import torch
import torch.optim as optim

def test(model, device, test_loader, criterion):
    model.eval()

    # accuracy stats collection
    correct = 0
    processed = 0
    epoch_loss = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):
            image = batch['image'].to(device)
            question = batch['question'].to(device)
            target = batch['answer'].to(device)
            
            pred = model(image, question)
            loss = criterion(pred, target)

            # metrics
            predicted_digits = pred.argmax(dim=1, keepdim=True)
            correct += predicted_digits.eq(target.view_as(predicted_digits)).sum().item()
            processed += len(image)
            epoch_loss += loss.item()
    
    epoch_accuracy = (100 * correct) / processed
    epoch_loss /= len(test_loader)
    print(f"Inference-----Epoch Accuracy {round(epoch_accuracy, 2)}\tEpoch Loss {round(epoch_loss, 4)}")
    return epoch_accuracy, epoch_loss