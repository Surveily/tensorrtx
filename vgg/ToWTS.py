import torch
import struct
net = torch.load("model.pt")

f = open("vgg.wts", 'w')
f.write("{}\n".format(len(net.keys())))
for k,v in net.items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")
