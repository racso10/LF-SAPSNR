from torch.nn import init


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            # init.xavier_uniform_(m.bias.data, gain=0.02)
        except Exception as e:
            print(e)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.2)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
