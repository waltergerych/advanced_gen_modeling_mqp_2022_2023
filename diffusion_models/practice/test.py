import torch

labels = torch.tensor([1, 2, 2, 3, 4])

new_labels = torch.eq(labels, torch.ones(labels.size(0))*2).int()
print(new_labels)