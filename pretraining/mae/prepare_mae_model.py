import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import torch
import util.lr_decay as lrd

"""
Constructs a list of MAE layers which will remain unfrozen:

If we freeze the weights we always leave the final normalization layer ('fc_norm') and the final linear layer ('head') unfrozen.
Additionally, we also unfreeze the last 'number_of_blocks_to_leave_unfrozen' attention blocks of the ViT encoder.

Input:
    -> total_number_of_blocks (int): Total number of encoder attention blocks in the model.
    -> number_of_blocks_to_leave_unfrozen (int): Number of attention blocks to leave unfrozen in ViT encoder.

Output:
    -> weights_to_not_freeze (List): A list of layer names to leave unfrozen.
"""

def get_weights_to_not_freeze(total_number_of_blocks, number_of_blocks_to_leave_unfrozen):
    weights_to_not_freeze = ["head", "fc_norm"]

    for i in range(number_of_blocks_to_leave_unfrozen):
        weights_to_not_freeze = [f"blocks.{total_number_of_blocks - 1 - i}"] + weights_to_not_freeze

    return weights_to_not_freeze


"""
This function prepares a Vision Transformer model for finetuning.

Inputs:
    -> model_type (str): Which model to use. Options are: "vit_base_patch16", "vit_large_patch16" and "vit_huge_patch14" as
                      defined at the end of models_vit.py.
    
    -> nb_classes (int): The number of classes the final linear layer should output. Set this number to 0 to remove the final
                      linear layer (i.e. the "head" layer).
    
    -> drop_path (float): drop_path value of the ViT model. Default 0.1.
    
    -> pool_type ("str"): The output of final attention block of ViT encoder will have the following shape: [B, N, E],
                       where B=batch_size, N=number_of_patches, E=embedding_dimension. This output first gets pooled and then
                       forwarded to the final linear layer ("head"). 
                       If pool_type=None no pooling is applied. That is we output per-patch embeddings.
                       If pool_type="global_pool" the final image embedding will be the mean of all the patch embeddings.
                       If pool_type="cls_token" the ViT cls token is used as the final image embedding.
    
    -> mae_ckpt (str): Path to the MAE checkpoint to initialize the weights with.
    
    -> freeze_weights (int):
        Options: -1 - all weights are unfrozen
                  0 - only the final linear layer is unfrozen (i.e. the 'head' layer)
                  1< i <n - freeze everything except last i attention blocks and the final linear layer (n is the number of encoder 
                            attention blocks)
    
    -> reinit_n_layers (int):
        Options: <= 0 - Don't re-initialize any layers
                 1< i <n - Re-initialize the head and fc_norm layers as well as last i attention blocks of the encoder.
                           (In the attention blocks we re-initialize linear layers and layer norm layers.)
                           n = number of encoder attention blocks.
    
    -> return_optimizer_groups (bool): Whether or not to prepare a list of MAE model parameter groups which you can then pass to the optimizer
                                    of your training procedure. The output will be a list of dictionaries with three keys:
                                    1) "params": a list of model parameters belonging to this group
                                    2) "weight_decay": weight decay for the group
                                    3) "lr_scale": scaling to apply to the LR of your training procedure

                                    This automatically sets weight decay for all the parameters (except for linear layer biases and layer 
                                    norm layers). It also applies Layer Wise Learning Rate Decay (LLRD), by calculating a LR scaling which
                                    should be applied to the learning rate before training.
                                    
                                    NOTE: In order to train your model properly, before passing the parameter groups to the optimizer, you 
                                    should add a "lr" key to each group and set it's value:

                                    for g in optimizer_groups:
                                        g["lr"] = your_initial_LR * g["lr_scale"]

    -> weight_decay (float): Weight decay to apply in case return_optimizer_groups==True.
    
    -> layer_decay (float): Layer Wise Learning Rate decay to apply in case return_optimizer_groups==True.
    
    -> verbose (bool): Prints out the prepared model, as well as total number of trainable parameters.
                       Moreover, will print out all the parameters that have requires_grad == False.

    -> debug (bool): Use this flag to check whether re-initialization was done correctly. This will print out weights and 
                     biases of the re-initialized parameters. Weights should be set to 1.0 and biases to 0.0.
                 
"""

def prepare_mae_model(model_type, nb_classes, drop_path, pool_type, mae_ckpt, freeze_weights, reinit_n_layers=-1, return_optimizer_groups=False, weight_decay=0., layer_decay=1.0, verbose=True, debug=False):
    # build model
    print("Configuring MAE model:")

    model = models_vit.__dict__[model_type](
        num_classes=nb_classes,
        drop_path_rate=drop_path,
        pool_type=pool_type
    )

    # re-init last n layers of encoder
    if (model.reinit_possible(reinit_n_layers)):
        model.reinit_weights(reinit_n_layers)

        print(f"Re-initialized {reinit_n_layers} layers.")

        if (debug):
            model._debug_reinit("RE-INIT DEBUG:")

    # load a pre-trained model
    if (mae_ckpt):
        checkpoint = torch.load(mae_ckpt, map_location='cpu')

        print("Loaded pre-trained MAE checkpoint from: %s" % mae_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and (isinstance(model.head, torch.nn.Identity) or checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if hparams.global_pool:
        #    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    if(freeze_weights != -1):
        print("Freezing all layers, except: ")
        wtnf = get_weights_to_not_freeze(len(model.blocks), freeze_weights)

        for name, param in model.named_parameters():
            if(any(map(lambda prefix: name.startswith(prefix), wtnf))):
                print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False

    # manually initialize fc layer
    if (not model.reinit_possible(reinit_n_layers) and not isinstance(model.head, torch.nn.Identity)):
        trunc_normal_(model.head.weight, std=2e-5)

    # set up LLRD and weight decay
    param_groups = None
    if (return_optimizer_groups):
        param_groups = lrd.param_groups_lrd(model, weight_decay,
                                            no_weight_decay_list={'pos_embed', 'cls_token'},
                                            layer_decay=layer_decay, verbose=True)

    n_parameters = sum(p.numel() for p in model.parameters())
    n_parameters_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if (verbose):
        print("*" * 200)
        print("*" * 200)
        print("")
        print("MAE Model = %s" % str(model))
        print("")
        print('number of params (M): %.2f' % (n_parameters / 1.e6))
        print('number of trainable params (M): %.2f' % (n_parameters_train / 1.e6))
        print("")
        print("*" * 200)
        print("*" * 200)
        
        # Printing non-trainable parameters.
        print("")
        print("Listing all parameters which have requires_grad == False:")
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"\t{name}")

        print("")
        print("*" * 200)
        print("*" * 200)
        print("")

    print('MAE model loaded.')

    return model if not return_optimizer_groups else {"model": model, "param_groups": param_groups} # 2 different outputs for compability reasons with TeCNO code