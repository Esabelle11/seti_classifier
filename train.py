import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score, classification_report
from dataset import SetiDataset, get_class_labels
from model import VGG16

# 路径根据Kaggle环境调整
train_path = '/kaggle/input/seti-data/primary_small/train'
val_path = '/kaggle/input/seti-data/primary_small/valid'
test_path = '/kaggle/input/seti-data/primary_small/test'

train_dataset = SetiDataset(train_path, train=True)
val_dataset = SetiDataset(val_path, train=False)
test_dataset = SetiDataset(test_path, train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train(model, train_loader, test_loader, optimizer, model_name, num_epochs=10, p=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    train_loss_set, test_loss_set = [], []
    acc_train_set, acc_test_set = [], []
    best_test_loss = float('inf')
    train_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_train_labels, all_train_outputs = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_train_labels.extend(labels.cpu().detach().numpy())
            outputs = outputs.argmax(dim=1)
            all_train_outputs.extend(outputs.cpu().detach().numpy())
        average_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_outputs)
        train_loss_set.append(average_train_loss)
        acc_train_set.append(train_accuracy)
        model.eval()
        all_test_labels, all_test_preds = [], []
        with torch.no_grad():
            total_test_loss = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(preds.cpu().numpy())
                total_test_loss += criterion(outputs, labels).item()
            average_test_loss = total_test_loss / len(test_loader)
            test_loss_set.append(average_test_loss)
            test_accuracy = accuracy_score(all_test_labels, all_test_preds)
            acc_test_set.append(test_accuracy)
            current_lr = optimizer.param_groups[0]['lr']
            if average_test_loss < best_test_loss:
                best_test_loss = average_test_loss
                torch.save(model.state_dict(), f'/kaggle/working/model_weights_{model_name}.pth')
            if epoch % p == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}: Lr:{current_lr}, train loss: {average_train_loss:.4f} train_acc:{train_accuracy:.4f} test_loss:{average_test_loss:.4f}, test acc:{test_accuracy:.4f}')
                print('-' * 80)
    train_end = time.time()
    time_use = train_end - train_start
    print(f'Time used for training:{time_use} sec')
    print('-' * 80)
    report_train = classification_report(all_train_labels, all_train_outputs, target_names=get_class_labels())
    report_test = classification_report(all_test_labels, all_test_preds, target_names=get_class_labels())
    print(f'Training Classification Report :\n {report_train}')
    print(f'Test Classification Report :\n {report_test}')
    return model, train_loss_set, test_loss_set, acc_train_set, acc_test_set, time_use

if __name__ == '__main__':
    vgg16 = VGG16()
    model_name = 'vgg16pre'
    num_epochs = 10
    p = 1
    optimizer = optim.Adam(vgg16.parameters(), lr=0.0001, weight_decay=0.001)
    model_VGG16, train_loss, test_loss, acc_train, acc_test, time_use = train(vgg16, train_loader, val_loader, optimizer=optimizer, model_name=model_name, num_epochs=num_epochs, p=p)
    # 保存完整模型
    torch.save(vgg16, '/kaggle/working/model_full_vgg16pre.pth')
    print('训练与模型保存完成')
