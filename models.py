#Modified https://github.com/pytorch/examples/blob/master/mnist/main.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    #Note that the input to a conv layer should be a tensor of size (batch_size,num_channels, height, width)
    #Though batch size in dataloader can be changed independently... why?
    #https://discuss.pytorch.org/t/how-to-define-the-kernel-size-for-conv2d/21777
    #self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv1=nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1)
    self.conv2=nn.Conv2d(20, 50, 5, 1)
    self.fc1=nn.Linear(50*5*5,500)
    self.fc2=nn.Linear(500,10)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 5*5*50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
