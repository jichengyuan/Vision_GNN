import matplotlib.pyplot as plt
import math
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2
from torch.autograd import Function


pre_transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),                 
                                 transforms.Resize((224, 224))])

post_transforms = transforms.Compose([transforms.Normalize(mean = [0, 0, 0], std = [1/0.229, 1/0.224, 1/0.225]),                 
                              transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1, 1, 1])])

pre_transforms_gray = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.449], std = [0.226]),                 
                                 transforms.Resize((224, 224))])

post_transforms_gray = transforms.Compose([transforms.Normalize(mean = [0], std = [1/0.226]),                 
                              transforms.Normalize(mean = [-0.449], std = [1])])

def scale(arr):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

def preprocess_img(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Resize((224, 224))
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def preprocess_image(img_path, rgb = True):
    if isinstance(img_path, str):
        img = Image.open(img_path)
    else:
        img = img_path
    
    if rgb:
        img_t = pre_transforms(img).unsqueeze(0)
    else:
        img_t = pre_transforms_gray(img).unsqueeze(0)
    return img_t

def postprocess_image(img_t, rgb = True):
    if rgb:
        img_t = post_transforms(img_t[0])
    else:
        img_t = post_transforms_gray(img_t[0])        
        
    img_np = img_t.detach().cpu().numpy().transpose(1,2,0)
    img_np = scale(np.clip(img_np, 0, 1))
    
    return img_np

def visualize_filters(model, filter_name = None, max_filters = 64, size = 128, figsize = (16, 16), save_path = None):
    """
        Plots filters of a convolutional layer by interpreting them as grayscale images
    """

    name, weights = next(model.named_parameters())
    
    for layer_name, layer_weights in model.named_parameters():
        if layer_name == filter_name:
            name = layer_name
            weights = layer_weights 
          
    w_size = weights.size()
    # print(f'Visualizing ')
    merged_weights = weights.reshape(w_size[0] * w_size[1], w_size[2], w_size[2]).detach().cpu().numpy()
    out_chs = merged_weights.shape[0]
    
    if out_chs > max_filters:
        merged_weights = merged_weights[torch.randperm(out_chs)[:max_filters]]
        out_chs = max_filters    
    
    sqrt = int(math.sqrt(out_chs))
    fig, axs = plt.subplots(sqrt, sqrt, figsize = figsize)
    
    if not size:
        size = merged_weights.shape[2]
    
    for i in range(sqrt ** 2):
        weight = merged_weights[i]
        scaled = scale(weight)
        resized = transforms.Resize((size, size))(Image.fromarray(scaled))
        plot_idx = int(i / sqrt), i % sqrt
        
        axs[plot_idx].imshow(resized, cmap = 'gray')
        axs[plot_idx].set_yticks([])
        axs[plot_idx].set_xticks([])
    
    if save_path:
        fig.savefig(save_path)


def visualize_activations(model, module, img_path, max_acts = 64, rgb = True, figsize = (16, 16), save_path = None, device='gpu'):
    """
        Plots the activations of a module recorded during a forward pass on an image
    """

    img_t = preprocess_image(img_path, rgb = rgb)
    acts = [0]

    def hook_fn(self, input, output):
        acts[0] = output
    
    handle = module.register_forward_hook(hook_fn)
    if device == 'gpu':
        model.to('cuda:0')
        img_t = img_t.to('cuda:0')
    out = model(img_t)
    handle.remove()
    if device == 'gpu':
        acts = acts[0][0].cpu().detach().numpy()
    else:
        acts = acts[0][0].detach().numpy()
    
    if acts.shape[0] > max_acts:
        acts = acts[torch.randperm(acts.shape[0])[:max_acts]]
    
    sqrt = int(math.sqrt(acts.shape[0]))
    fig, axs = plt.subplots(sqrt, sqrt, figsize = figsize)
    
    for i in range(sqrt ** 2):
        scaled = scale(acts[i])
        
        plt_idx = int(i / sqrt), i % sqrt
        axs[plt_idx].imshow(scaled, cmap = 'gray')
        axs[plt_idx].set_yticks([])
        axs[plt_idx].set_xticks([])
    if save_path:
        fig.savefig(save_path)
        

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x
    
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x
    
class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())
        # print(output.size())
        output = output.permute(0, 2, 3, 1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # print(one_hot.shape)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        temp = one_hot * output
        # print("temp: ", temp.size())
              
        one_hot = torch.sum(one_hot * output)
        
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
    
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())
        # output = output.permute(0, 2, 3, 1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        # print(one_hot.size())
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)
