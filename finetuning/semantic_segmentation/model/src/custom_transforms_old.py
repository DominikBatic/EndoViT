import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

########################################################################################################################
################################################ CUSTOM TRANSFORMATIONS ################################################
########################################################################################################################

class custom_RandomResizedCrop(object):
    def __init__(self, size=[224, 224], scale=[0.08, 1.0], ratio=[3.0 / 4.0, 4.0 / 3.0]):
        assert isinstance(size, (int, list))
        if (isinstance(size, int)):
            self.size = [size, size]
        else:
            assert len(size) == 2, f"Argument \"size\" should be a list of 2 parameters. Instead got \"{len(size)}\" parameters."
            self.size = size

        assert isinstance(scale, list), f"Argument \"scale\" should be a list, instead got {type(scale)}."
        assert len(scale) == 2, f"Argument \"scale\" should be a list of 2 parameters. Instead got \"{len(scale)}\" parameters."
        self.scale = scale

        
        assert isinstance(ratio, list), f"Argument \"ratio\" should be a list, instead got {type(ratio)}."
        assert len(ratio) == 2, f"Argument \"ratio\" should be a list of 2 parameters. Instead got \"{len(ratio)}\" parameters."
        self.ratio = ratio

    def __call__(self, sample):
        image, ground_truth, color_mask = sample["image"], sample["gt_segmentation"], sample["gt_color_mask"]

        assert image.size == ground_truth.size, f"Incompatible shapes: image {image.size} <--> gt_segmentation {ground_truth.size}"
        assert image.size == color_mask.size,   f"Incompatible shapes: image {image.size} <--> gt_color_mask {color_mask.size}"
        
        i, j, h, w = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)

        image        = F.resized_crop(image       , i, j, h, w, self.size, T.InterpolationMode.BICUBIC)
        ground_truth = F.resized_crop(ground_truth, i, j, h, w, self.size, T.InterpolationMode.NEAREST)
        color_mask   = F.resized_crop(color_mask  , i, j, h, w, self.size, T.InterpolationMode.NEAREST)

        return {"image": image, "gt_segmentation": ground_truth, "gt_color_mask": color_mask}
    
    def __str__(self):
        return f"custom_Resize(size={self.size}, scale={self.scale}, ratio={self.ratio})"


class custom_Resize(object):
    def __init__(self, size=(224, 224)):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2, f"Argument \"size\" should be a tuple of 2 parameters. Instead got \"{len(size)}\" parameters."
            self.size = size

        self.resize_image = T.Resize(size=self.size, interpolation=T.InterpolationMode.BICUBIC)
        self.resize_segmentation_masks = T.Resize(size=self.size, interpolation=T.InterpolationMode.NEAREST)

    def __call__(self, sample):
        image, ground_truth, color_mask = sample["image"], sample["gt_segmentation"], sample["gt_color_mask"]

        image        = self.resize_image(image)
        ground_truth = self.resize_segmentation_masks(ground_truth)
        color_mask   = self.resize_segmentation_masks(color_mask)

        return {"image": image, "gt_segmentation": ground_truth, "gt_color_mask": color_mask}
    
    def __str__(self):
        return f"custom_Resize(size={self.size})"


class custom_RandomHorizontalFlip(object):
    def __init__(self, probability):
        assert isinstance(probability, (int, float))
        self.flip_probability = float(probability)

    def __call__(self, sample):
        image, ground_truth, color_mask = sample["image"], sample["gt_segmentation"], sample["gt_color_mask"]

        random_number = torch.rand(1).item()

        if (random_number < self.flip_probability):
            image = F.hflip(image)
            ground_truth = F.hflip(ground_truth)
            color_mask = F.hflip(color_mask)

        return {"image": image, "gt_segmentation": ground_truth, "gt_color_mask": color_mask}
    
    def __str__(self):
        return f"custom_RandomHorizontalFlip(flip_probability={self.flip_probability})"
    

class custom_ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, ground_truth, color_mask = sample["image"], sample["gt_segmentation"], sample["gt_color_mask"]

        image = F.to_tensor(image) #.to(torch.float)
        ground_truth = F.to_tensor(ground_truth) #.to(torch.long)
        color_mask = F.to_tensor(color_mask) #.to(torch.long)

        return {"image": image, "gt_segmentation": ground_truth, "gt_color_mask": color_mask}
    
    def __str__(self):
        return f"custom_ToTensor()"

  
class custom_Normalize(object):
    def __init__(self, mean, std):
        assert isinstance(mean, list) and len(mean) == 3, f"Mean of the dataset should be a list of 3 parameters. Instead got \"{len(mean)}\" parameters."
        assert isinstance(std, list) and len(std) == 3, f"Standard deviation of the dataset should be a list of 3 parameters. Instead got \"{len(std)}\" parameters."
        self.dataset_mean = mean
        self.dataset_std = std

    def __call__(self, sample):
        image, ground_truth, color_mask = sample["image"], sample["gt_segmentation"], sample["gt_color_mask"]
            
        image = F.normalize(image, self.dataset_mean, self.dataset_std)

        return {"image": image, "gt_segmentation": ground_truth, "gt_color_mask": color_mask}
    
    def __str__(self):
        return f"custom_Normalize(mean={self.dataset_mean}, std={self.dataset_std})"



#---------------------------
# Functions
#---------------------------

# Not used.
def build_custom_transform(transform_hyperparams_dict):

    transform_dictionary = {
        "RandomResizedCrop": custom_RandomResizedCrop,
        "RandomHorizontalFlip": custom_RandomHorizontalFlip,
        "Resize": custom_Resize,
        "ToTensor": custom_ToTensor,
        "Normalize": custom_Normalize,
    }

    transform = T.Compose([transform_dictionary[transform](*hyperparams) for transform, hyperparams in transform_hyperparams_dict.items()])
    
    return transform