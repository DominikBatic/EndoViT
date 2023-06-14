import torch

# Dictionary mapping watershed_mask colors to their corresponding classes.
# The colors are in the following format: 
#       #{RGB value 0 to 255}{RGB value 0 to 255}{RGB value 0 to 255}

# NOTE: This is the official dictionary given at the CholecSeg8k kaggle webpage.
#       However,
#       while processing images we noticed there are two extra colors in the images.
#       1) Watershed masks as well as Color masks have a white 1-pixel-wide edge.
#       2) Some pixels in the images have not been segmented properly and their masks
#          are missing. These pixels are colored black in watershed masks and white in
#          color masks.
#       Because of this we will use a modified version of this dictionary listed below.

# Not used in code.
watershed_to_class = {
    "#505050":  0, # Black Background
    "#111111":  1, # Abdominal Wall
    "#212121":  2, # Liver
    "#131313":  3, # Gastrointestinal Tract
    "#121212":  4, # Fat
    "#222222":  5, # Gallbladder
    # ----- Combined into 1 class -----
    "#232323":  6, # Connective Tissue
    "#242424":  6, # Blood
    "#252525":  6, # Cystic Duct
    "#333333":  6, # Hepatic Vein
    "#050505":  6, # Liver Ligament
    # ----- Combined into 1 class -----
    "#313131":  7, # Grasper
    "#323232":  7, # L-hook Electrocautery
}

# MODIFIED dictionary mapping watershed_mask colors to their corresponding classes.
# The colors are in the following format: 
#       #{RGB value 0 to 255}{RGB value 0 to 255}{RGB value 0 to 255}

# The last two entries are colors/classes that shouldn't appear in the ground truth
# masks. What we decided to do is to map them to a new 9th class (number 8).
# We will make the model output logits for all 9 classes, however when calculating
# the loss and metric values we will ignore the new class by setting ignore_index=8.

# Not used in code.
watershed_to_class_v2 = {
    "#505050":  0, # Black Background
    "#111111":  1, # Abdominal Wall
    "#212121":  2, # Liver
    "#131313":  3, # Gastrointestinal Tract
    "#121212":  4, # Fat
    "#222222":  5, # Gallbladder
    # ----- Combined into 1 class -----
    "#232323":  6, # Connective Tissue
    "#242424":  6, # Blood
    "#252525":  6, # Cystic Duct
    "#333333":  6, # Hepatic Vein
    "#050505":  6, # Liver Ligament
    # ----- Combined into 1 class -----
    "#313131":  7, # Grasper
    "#323232":  7, # L-hook Electrocautery
    # ----- Combined into 1 class -----
    # These should not be in the dataset, but there were some errors in labelling.
 "#255255255":  8,
    "#000000":  8,
}

# Dictionary mapping watershed_mask colors to their corresponding classes.
# The mapping is 1 x RGB_color_value -> torch.FloatTensor of 3 x class_number

# This version is used in the code.
watershed_to_class_v3 = {
    "50": torch.FloatTensor([ 0,  0,  0]), # Black Background
    "11": torch.FloatTensor([ 1,  1,  1]), # Abdominal Wall
    "21": torch.FloatTensor([ 2,  2,  2]), # Liver
    "13": torch.FloatTensor([ 3,  3,  3]), # Gastrointestinal Tract
    "12": torch.FloatTensor([ 4,  4,  4]), # Fat
    "22": torch.FloatTensor([ 5,  5,  5]), # Gallbladder
    # ----- Combined into 1 class -----
    "23": torch.FloatTensor([ 6,  6,  6]), # Connective Tissue
    "24": torch.FloatTensor([ 6,  6,  6]), # Blood
    "25": torch.FloatTensor([ 6,  6,  6]), # Cystic Duct
    "33": torch.FloatTensor([ 6,  6,  6]), # Hepatic Vein
    "05": torch.FloatTensor([ 6,  6,  6]), # Liver Ligament
    # ----- Combined into 1 class -----
    "31": torch.FloatTensor([ 7,  7,  7]), # Grasper
    "32": torch.FloatTensor([ 7,  7,  7]), # L-hook Electrocautery
    # ----- Combined into 1 class -----
    # These should not be in the dataset, but there were some errors in labelling.
   "255": torch.FloatTensor([ 8,  8,  8]),
    "00": torch.FloatTensor([ 8,  8,  8]),
}

# Dictionary mapping classes to their corresponding color in color_masks.

# This version is used in the code to create visualization images out of the predictions.
class_to_color = {
     '0': torch.FloatTensor([127, 127, 127]) / 255., # Black Background
     '1': torch.FloatTensor([210, 140, 140]) / 255., # Abdominal Wall
     '2': torch.FloatTensor([255, 114, 114]) / 255., # Liver
     '3': torch.FloatTensor([231,  70, 156]) / 255., # Gastrointestinal Tract
     '4': torch.FloatTensor([186, 183,  75]) / 255., # Fat
     '5': torch.FloatTensor([255, 255,   0]) / 255., # Gallbladder
    # ----- Combined into 1 class: "Misc" class -----
     '6': torch.FloatTensor([ 45, 158, 227]) / 255., # Connective Tissue + Blood + Cystic Duct + Hepatic Vein + Liver Ligament
    # ----- Combined into 1 class: "Instruments" class -----
     '7': torch.FloatTensor([170, 255,   0]) / 255., # Grasper + L-hook Electrocautery
    # ----- Combined into 1 class: "Erros class" -----
     '8': torch.FloatTensor([255, 255, 255]) / 255., # White border + black/white pixels.
}

# Dictionary mapping watershed_mask colors to their corresponding color in color_masks.

# Not used in code.
watershed_to_color = {
    '#505050': torch.FloatTensor([127, 127, 127]) / 255., # Black Background
    '#111111': torch.FloatTensor([210, 140, 140]) / 255., # Abdominal Wall
    '#212121': torch.FloatTensor([255, 114, 114]) / 255., # Liver
    '#131313': torch.FloatTensor([231,  70, 156]) / 255., # Gastrointestinal Tract
    '#121212': torch.FloatTensor([186, 183,  75]) / 255., # Fat
    '#313131': torch.FloatTensor([170, 255,   0]) / 255., # Grasper
    '#232323': torch.FloatTensor([255,  85,   0]) / 255., # Connective Tissue
    '#242424': torch.FloatTensor([255,   0,   0]) / 255., # Blood
    '#252525': torch.FloatTensor([255, 255,   0]) / 255., # Cystic Duct
    '#323232': torch.FloatTensor([169, 255, 184]) / 255., # L-hook Electrocautery
    '#222222': torch.FloatTensor([255, 160, 165]) / 255., # Gallbladder
    '#333333': torch.FloatTensor([  0,  50, 128]) / 255., # Hepatic Vein
    '#050505': torch.FloatTensor([111,  74,   0]) / 255., # Liver Ligament
 '#255255255': torch.FloatTensor([255, 255, 255]) / 255., # White color in watershed mask is the 1 pixel border
    '#000000': torch.FloatTensor([255, 255, 255]) / 255., # Black pixels are errors in the segmentation. The same pixels are white in color masks.
}
