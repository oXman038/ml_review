import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


def print_model(model,
                model_desc=True,
                named_parameters=False,
                short=False):
  if model_desc:
    if short:
      print('\n'.join(model.__str__().split('\n')[:30]))
    else:
      print(model)

  if named_parameters:
    if short:
      for idx, (name, param) in enumerate(model.named_parameters()):
        print("%30s | %s" % (name, param.size()))
        if idx>=20:
          break
    else:
      for name, param in model.named_parameters():
        print("%30s | %s" % (name, param.size()))


def exp_derivative():
  a = torch.tensor([2., 3.], requires_grad=True)
  b = torch.tensor([6., 4.], requires_grad=True)
  Q = 3*a**3 - b**2
  print(Q)

  # external_grad = torch.tensor([2., 2.])
  # Q.backward(gradient=external_grad)
  Q.sum().backward()

  print(9*a**2)
  print(a.grad)

  print(-2*b)
  print(b.grad)

  print(9*a**2 == a.grad)
  print(-2*b == b.grad)


def exp_learning_rate_check():
  model = torchvision.models.resnet18(pretrained=True)
  data = torch.rand(1, 3, 64, 64)
  label = torch.rand(1, 1000)
  print_model(model, model_desc=False,
              named_parameters=True, short=False)

  prediction = model(data)
  loss = (prediction - label).sum()
  loss.backward()
  grad_values = model.fc.weight.grad.clone().detach()
  print(loss)
  print("Gradients of last layer:")
  print(grad_values)

  pre_values = model.fc.weight.clone().detach()
  optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
  optim.step()
  cur_values = model.fc.weight.clone().detach()

  diff_values = (cur_values - pre_values)
  print("Last layer updates via SGD")
  print(diff_values)

  print("Update ratio:")
  print(diff_values/grad_values)

  expected_values = torch.ones_like(diff_values).fill_(-0.01)
  print(torch.isclose((diff_values/grad_values), expected_values))


if __name__ == "__main__":
  # Get cpu or gpu device for training.
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Using {} device".format(device))

  # exp_derivative()
  exp_learning_rate_check()
