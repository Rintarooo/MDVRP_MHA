import torch

# n_depot = 2
# batch = 5
# a = torch.arange(2).reshape(1,-1).repeat(batch,1)
# # c = torch.tensor([1])
# c = torch.randint(low = 0, high = 5, size = (batch, 1))
# print(a)
# print(a.size())
# print(c.size())
# print(c)
# bool = (c == a).sum(-1).bool()
# print(bool.size())
# print(bool)
# # print(bool.sum(1))

n_depot = 2
batch = 5
n_car = 3
n_customer = 4
next_car = torch.randint(low = 0, high = n_car, size = (batch, 1))
is_next_depot = torch.randint(low = 0, high = 2, size = (batch, 1)).bool()
demand = torch.rand((batch, 1))
print(demand)
print(next_car)
one_hot = torch.eye(n_car)[next_car].reshape(batch, n_car)
print(one_hot)
print(one_hot.size())# (batch, n_car)
print(next_car.size())
c = torch.logical_not(is_next_depot).long() * one_hot * demand#.expand(batch, n_car)
print(c)
print(is_next_depot)
a = torch.ones((batch, n_car))
print(a)
a -= c
print(a)

import torch

batch = 10
n_car = 3
next_car = torch.randint(low = -2, high = n_car, size = (batch, 1))
print(next_car)

next_car = torch.clamp(next_car, min = 0, max = n_car)
print(next_car)

a = torch.ones((batch, n_car))
print(a)
b = torch.logical_not(a)
print(b)
