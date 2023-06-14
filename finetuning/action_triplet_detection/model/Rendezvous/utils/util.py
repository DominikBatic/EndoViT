# track learning rates in the mae model
def collect_lr_by_layers(optimizer):    
    lr = []
    lr.append(optimizer.param_groups[0]["lr"]) # layer0 embeddings
    lr.append(optimizer.param_groups[2]["lr"]) # layer1
    lr.append(optimizer.param_groups[4]["lr"]) # layer2
    lr.append(optimizer.param_groups[6]["lr"]) # layer3
    lr.append(optimizer.param_groups[8]["lr"]) # layer4
    lr.append(optimizer.param_groups[10]["lr"]) # layer5
    lr.append(optimizer.param_groups[12]["lr"]) # layer6
    lr.append(optimizer.param_groups[14]["lr"]) # layer7
    lr.append(optimizer.param_groups[16]["lr"])  # layer8
    lr.append(optimizer.param_groups[18]["lr"])  # layer9
    lr.append(optimizer.param_groups[20]["lr"])  # layer10
    lr.append(optimizer.param_groups[22]["lr"])  # layer11
    lr.append(optimizer.param_groups[24]["lr"])  # layer12
    lr.append(optimizer.param_groups[26]["lr"])  # layer13 head and fc_norm 
    return lr