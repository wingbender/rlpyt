import torch
import csv
temp = torch.rand((3,5))
filename = '../data/temp.pt'
torch.save(temp,filename)
temp2 = torch.load(filename)
assert torch.allclose(temp,temp2)
with open('../data/tensor.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(temp.tolist())
print(temp2)
print('Test success')