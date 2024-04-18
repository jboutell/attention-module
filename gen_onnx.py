from MODELS.model_resnet import *
import onnx
import os

# Read Pretrained model
model = ResidualNet('ImageNet', 50, 1000, 'CBAM')
optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=1e-4)
model = torch.nn.DataParallel(model, device_ids=list(range(1)))
saved_model = os.path.join("SAVED_MODEL","RESNET50_CBAM_new_name_wrap.pth")
checkpoint = torch.load(saved_model)
model.load_state_dict(checkpoint['state_dict'])

img = torch.randn(1, 3, 224, 224, device="cuda")
model_name = 'cbam-resnet50.onnx'
torch.onnx.export(model.module.cuda(), img, model_name, opset_version=13, verbose=False, input_names = ['img'], output_names = ['logits'])

