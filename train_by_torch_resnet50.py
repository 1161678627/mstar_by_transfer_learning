import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import copy
import os

# 1.读取数据及预处理
data_dir = {'train': './train', 'valid': './test'}

# 数据预处理以及数据增强操作
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(200),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),     # 图像随机反转
        transforms.RandomVerticalFlip(p=0.5),   # 图像垂直反转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 随机的对图片的亮度、对比度等进行调整
        transforms.RandomGrayscale(p=0.025),    # 随机的将图片转为灰度-但依旧保持三通道，只是三个通道的像素数值都相等
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(200),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# 生成datasets和dataloaders对象
image_datasets = {x: datasets.ImageFolder(data_dir[x], transform=data_transforms[x]) for x in ['train', 'valid']}
classes_name = image_datasets['train'].classes
print(classes_name)

# 指定dataloaders每次加载数据的batch数量，冻结预训练层时可以设置的较大
batch_size = 16
image_dataloaders = {x: DataLoader(dataset=image_datasets[x], batch_size=batch_size, shuffle=True, drop_last=True) for x in ['train', 'valid']}

# 开始配置迁移学习的model
model = models.resnet50(pretrained=True)
# 将所有层都设为不可训练
for param in model.parameters():
    # param.requires_grad = False
    param.requires_grad = True

# 修改最后一层
num_features = model.fc.in_features
classes_num = 10
# print(num_features)  # 2048
# print(model)
model.fc = torch.nn.Sequential(torch.nn.Linear(num_features, 256),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(p=0.5),
                               torch.nn.Linear(256, classes_num),
                               torch.nn.LogSoftmax(dim=1))

# 查看修改model后的 param 可训练情况，并将其记录以便传入 优化器
params_to_update = []
for param in model.parameters():
    # print(param.requires_grad)
    if param.requires_grad:
        params_to_update.append(param)



# 配置优化器和loss函数
learning_rate = 1e-3
optimizer = torch.optim.Adam(params=params_to_update, lr=learning_rate)
# 配置学习率下降方式，一开始学习率较大，随着epoch进行，学习率逐渐减小，每7个epoch衰减成原来的1/10---加快优化速度和效果
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

# 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = torch.nn.NLLLoss()

# 给当前model配置gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('train on gpu!')
else:
    print('train on cpu!')
model = model.to(device)

# model checkpoint save
checkpoint_path = './checkpoint/checkpoint.pth'

# tensorboardX log save path
log_save_path = './log/'
writer = SummaryWriter(logdir='./logs/train_log', )

epochs = 100

def train_model(model, dataloaders, criterion, optimizer,scheduler, epochs, checkpoint_filename, writer):
    since = time.time()
    best_acc = 0

    # 记录训练中每个epochs的 acc 和 loss 以便观察
    train_acc_history = []
    val_acc_history = []
    train_losses = []
    val_losses = []

    LRs = []

    # 如果之前训练过，可以通过判断 checkpoint文件在不在决定是否 续着训练
    if os.path.exists(checkpoint_filename):
        print('load the checkpoint!')
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint['state_dict'])

        # 因为此时我们训练了所有层，所以当前 优化器中待训练的参数 和 checkpoint中的 数量是不一致的，此时无法加载优化器，只能重新设置一个优化器
        # optimizer.load_state_dict(checkpoint['optimizer'])

        best_acc = checkpoint['best_acc']
        # model.class_to_idx = checkpoint['mapping']

        # 如果不是加载历史的 model的优化器 ，optimizer.param_group[0]['lr']为空会报错
        # LRs = [optimizer.param_group[0]['lr']]

        print(best_acc, LRs)

    model.to(device)

    # 预定义一个当前最佳model的内存对象---在训练循环中更替
    best_model_wts = copy.deepcopy(model.state_dict())

    # 开始进入正式训练流程
    for epoch in range(epochs):
        print(f'epoch {epoch + 1}/{epochs}')
        print('-' * 20)

        # 每一个epoch 都包括 train 和 valid 两个过程，指定train或者eval 主要是为了BN和Dropout层 在训练和测试 时候有所不同，需要说明
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 评估

            # 用于统计每个 训练 epoch 中的 loss 和 corrects
            running_loss = 0.0
            runing_corrects = 0.0

            # 开始进入真实的取数据-训练 循环
            # 使用for循环 从 dataloader 中取数据，每次取出指定的一个 batch size
            for inputs, labels in dataloaders[phase]:

                # 将输入转到gpu中
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 梯度清零
                optimizer.zero_grad()

                # 计算梯度并反向传播---->尽在训练时进行
                # torch.set_grad_enabled(mode) 当 mode=True 时对with下的操作记录梯度，否则不记录梯度，训练时开始记录梯度，验证时=False
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # 对 outputs进行如下转换得到preds，才能和 labels的形式对应起来
                    _, preds = torch.max(outputs, 1)

                    # 仅在训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 累加每个 batch 的 损失
                running_loss += loss.item() * inputs.size(0)
                runing_corrects += torch.sum(preds == labels.data)

            # 迭代数据的for循环结束，标志着一个epoch训练结束，统计该epoch的平均 loss和acc
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = runing_corrects / len(dataloaders[phase].dataset)

            # 打印当前epoch的训练时间和准确率，损失的信息
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 每个epoch中 都需要在 valid 验证结束后，根据 valid 的 loss 和 acc 判断当前model是否当前最佳，并保存当前最佳的model
            if phase == 'valid' and epoch_acc > best_acc:
                # 更新最佳的 model 准确率
                best_acc = epoch_acc
                # 更新最佳的 model 训练参数 到 内存中，以便在训练结束后，直接从内存中加载最佳的model，而不用再从 checkpoint文件中去读取
                best_model_wts = copy.deepcopy(model.state_dict())
                # state中保存了完整的当前 model 的 checkpoint，主要用于以后恢复当前训练点，进行继续训练。
                # 如果只想保存当前model，以后用于预测任务，则仅需保存 model.state_dict()/best_model_wts 即可
                state = {
                    'state_dict': best_model_wts,  # model 每层权重参数
                    'best_acc': best_acc,  # 当前验证最佳准确率
                    'optimizer': optimizer.state_dict(),  # 当前训练过程中 优化器的参数
                    'classes_name': classes_name
                }
                torch.save(state, checkpoint_filename)

            # 记录每个 epoch 中的 train 的 acc 和 loss 变化数值，用于可视化训练信息
            if phase == 'train':
                scheduler.step()
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
                writer.add_scalar('train acc', scalar_value=epoch_acc, global_step=epoch+1)
                writer.add_scalar('train loss', scalar_value=epoch_loss, global_step=epoch + 1)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                val_losses.append(epoch_loss)
                writer.add_scalar('val acc', scalar_value=epoch_acc, global_step=epoch + 1)
                writer.add_scalar('val loss', scalar_value=epoch_loss, global_step=epoch + 1)

        # 当每个 epoch 训练完成后，记录当前epoch 优化器的学习率
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

        # 给日志文件中写入记录的变量
        writer.add_scalar('leraning rate', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch+1)


    # 当所有 epochs 训练完成后，打印训练花费的整体时间和 epoch最佳准确率
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 模型训练最后一轮的权重不一定是最佳权重，因此需要手动设置 训练过程中最佳的 权重 到model中
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, val_losses, train_acc_history, train_losses, LRs

# 仅训练新接的两层全连接层
model, val_acc_history, val_losses, train_acc_history, train_losses, LRs = train_model(model, image_dataloaders,
                                                criterion, optimizer, scheduler, epochs, checkpoint_path, writer)