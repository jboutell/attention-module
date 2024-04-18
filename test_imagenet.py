import torchvision.transforms as transforms
from MODELS.model_resnet import *
import torchvision.datasets as datasets
import time
import os
from train_imagenet import AverageMeter

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Read Pretrained model
model = ResidualNet('ImageNet', 50, 1000, 'CBAM')
optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=1e-4)
model = torch.nn.DataParallel(model, device_ids=list(range(1)))
saved_model = os.path.join("SAVED_MODEL","RESNET50_CBAM_new_name_wrap.pth")
checkpoint = torch.load(saved_model)
model.load_state_dict(checkpoint['state_dict'])

val_dir = os.path.join('data','ImageNet','val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=16, shuffle=False,
    num_workers=1, pin_memory=True)

# Evaluation loop
batch_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
print_freq = 10
criterion = nn.CrossEntropyLoss().cuda()
# switch to evaluate mode
model.eval()
end = time.time()
for i, (input, target) in enumerate(val_loader):
    target = target.cuda(non_blocking =True)
    with torch.no_grad():
      input_var = torch.autograd.Variable(input)
      target_var = torch.autograd.Variable(target)

    output = model(input_var, torch.Tensor([4]).cuda()) # can be adjusted in the range [0..4]
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses,
            top1=top1, top5=top5))

print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
      .format(top1=top1, top5=top5))

