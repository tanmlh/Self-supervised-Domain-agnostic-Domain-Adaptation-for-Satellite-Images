def check_freeze(net, net_conf):
    freeze = True if 'freeze' in net_conf and net_conf['freeze'] is True else False
    if freeze:
        for param in net.parameters():
            param.requires_grad = False
