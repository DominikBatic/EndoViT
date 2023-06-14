import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch

# get the CholecSeg8k color dict
import sys
from pathlib import Path
sys.path.append("./datasets/CholecSeg8k/utils")
import CholecSeg8k_color_dict_combined_classes as color_dict

#####################################################################
########################### Writing Utils ###########################
#####################################################################

def header(title="", separator="#", total_length=100, new_line=True, call_print=True):
    title = f"  {title}  " if title else title

    filler_total_len = (total_length - len(title))
    filler_left = filler_total_len // 2
    filler_right = filler_total_len - filler_left

    new_line = "\n" if new_line else ""
    to_print = ""

    to_print += new_line
    to_print += f"{separator * total_length}\n"
    to_print += f"{' ' * filler_left}{title}{' ' * filler_right}\n"
    to_print += f"{separator * total_length}\n"
    to_print += "\n"

    if (call_print):
        print(to_print, end="")
    
    return to_print


def new_section(title="", separator="*", total_length=50, new_line=True, call_print=True):
    title = f"  {title}  " if title else title

    filler_total_len = (total_length - len(title))
    filler_left = filler_total_len // 2
    filler_right = filler_total_len - filler_left

    new_line = "\n" if new_line else ""
    to_print = ""

    to_print += new_line
    to_print += f"{separator * filler_left}{title}{separator * filler_right}\n"
    to_print += "\n"

    if (call_print):
        print(to_print, end="")
    
    return to_print


def new_subsection(title="", separator="-", total_length=50, filler_length=5, new_line=True, call_print=True):
    title = f" {title} " if title else title

    empty_space_total_len = (total_length - 2*filler_length - len(title))
    empty_space_left = empty_space_total_len // 2
    empty_space_right = empty_space_total_len - empty_space_left

    new_line = "\n" if new_line else ""
    to_print = ""

    to_print += new_line
    to_print += f"{' ' * empty_space_left}{separator * filler_length}{title}{separator * filler_length}{' ' * empty_space_right}\n"
    to_print += "\n"

    if (call_print):
        print(to_print, end="")
    
    return to_print


def config_summary(config):

    header("CONFIG SUMMARY")

    header_length = 70
    to_print = ""

    for i, (group, params) in enumerate(config.items()):
        to_print += new_section(title=group, separator="*", total_length=header_length, new_line=True if i else False, call_print=False)
        
        for j, (subgroup_name, subgroup) in enumerate(params.items()):
            if(isinstance(subgroup, dict)):
                to_print += new_section(title=subgroup_name, separator="-", total_length=header_length, new_line=True if j else False, call_print=False)
                
                for hparam_name, hparam_value in subgroup.items():
                    to_print += "{0:<35s}: {1:35s}\n".format(hparam_name, str(hparam_value))
        
            else:
                hparam_name = subgroup_name
                hparam_value = subgroup
                to_print += "{0:<35s}: {1:35s}\n".format(hparam_name, str(hparam_value))
    
    print(to_print, end="")


def new_line(times=1):
    print("\n" * times, end="")




#####################################################################
########################### Plotting Utils ##########################
#####################################################################

def plot_image(axes, img, title=""):
    assert img.shape[2] == 3

    axes.imshow(img)
    axes.set_title(title)
    #plt.pause(0.001)
    
    return

def plot_grid(imgs, shape, title=""):
    assert len(shape) == 2
    
    fig = plt.figure(num="plot_grid", figsize=(shape[1] * 4, shape[0] * 4))
    
    fig.clf()
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    
    grid = ImageGrid(fig,
                     111,  # similar to subplot(111)
                     nrows_ncols=shape,
                     axes_pad=(0.2, 0.2),  # h / v padding
                 )
    
    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.set_axis_off()
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plot_image(ax, im)
        
    fig.suptitle(title, y=1.05)

    return fig


def create_blended_img(image, mask, alpha=0.5):
    """Blend the original image and segmentation mask.

    Args:
        image (array): input image
        mask (array): predicted segmentation mask
    """

    out = image * (1 - alpha) + mask * alpha

    # out.save(path + ".png")

    return out


def create_difference_image(gt_mask_3D, diff_mask, diff_color=torch.FloatTensor([0., 0., 238.]) / 255.):
    diff_mask_3D = diff_mask.repeat(3, 1, 1)
    out = gt_mask_3D * (1 - diff_mask_3D) +  diff_color.view([3, 1, 1]) * diff_mask_3D

    return out


def mask_to_color_img(mask):
    assert mask.ndim == 2

    original_shape = mask.shape
    
    mask = mask.reshape(-1)
    N_pixels = mask.shape[0]
    color_img = torch.zeros((3, N_pixels), dtype=torch.float)

    for i in range(N_pixels):
        color_img[:, i] = color_dict.class_to_color[str(mask[i].item())]

    color_img = color_img.reshape((3,) + tuple(original_shape))
    return color_img



def create_wandb_plots(config, input_img, gt_mask, pred):
    dataset_mean = torch.FloatTensor(config["Dataset"]["Transformations"]["dataset_mean"] or [0.3464, 0.2280, 0.2228])
    dataset_std  = torch.FloatTensor(config["Dataset"]["Transformations"]["dataset_std"]  or [0.2520, 0.2128, 0.2093])

    assert input_img.ndim == 3 and gt_mask.ndim == 2 and pred.ndim == 2
    assert input_img.shape[0] == 3

    # De-normalize the input.
    input_img = input_img * dataset_std.view([3, 1, 1]) + dataset_mean.view([3, 1, 1])

    # Turn gt_mask and prediction into colored 3-channel images.
    gt_mask_3D = mask_to_color_img(gt_mask)
    pred_3D = mask_to_color_img(pred)

    assert gt_mask_3D.shape == input_img.shape
    assert pred_3D.shape == input_img.shape

    # Which color to use as the error color.
    color = torch.FloatTensor([0., 0., 238.]) / 255. # blue

    # Create a difference image.
    diff_mask = gt_mask - pred
    diff_mask.to(torch.float)
    diff_mask = torch.where(diff_mask != 0., 1., 0.)

    diff_img = create_difference_image(gt_mask_3D, diff_mask, diff_color=color)

    assert diff_img.shape == input_img.shape

    # Overlay the images.
    # 1st overlay: input_img and gt_mask
    img_plus_gt = create_blended_img(input_img, gt_mask_3D)
    # 2nd overlay: input_img and prediction
    img_plus_pred = create_blended_img(input_img, pred_3D)

    # 3rd overlay: input_img and diff_img
    gt_plus_diff = create_blended_img(input_img, diff_img)
    # 4th overlay: input_img and the 3rd overlay
    #img_plus_gt_plus_diff = create_blended_img(input_img, 2*gt_plus_diff) * 2/3

    imgs_to_plot = [
        input_img, img_plus_gt, img_plus_pred, gt_plus_diff,
        input_img,  gt_mask_3D,       pred_3D,     diff_img
    ]
    imgs_to_plot = [img.permute(1, 2, 0).clamp(0., 1.) for img in imgs_to_plot]

    fig = plot_grid(imgs_to_plot, shape=(2, 4), title="")

    return fig