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
    "#313131":  5, # Grasper
    "#232323":  6, # Connective Tissue
    "#242424":  7, # Blood
    "#252525":  8, # Cystic Duct
    "#323232":  9, # L-hook Electrocautery
    "#222222": 10, # Gallbladder
    "#333333": 11, # Hepatic Vein
    "#050505": 12, # Liver Ligament
}

# MODIFIED dictionary mapping watershed_mask colors to their corresponding classes.
# The colors are in the following format: 
#       #{RGB value 0 to 255}{RGB value 0 to 255}{RGB value 0 to 255}

# The last two entries are colors/classes that shouldn't appear in the ground truth
# masks. What we decided to do is to map them to a new 14th class (number 13).
# We will make the model output logits for all 14 classes, however when calculating
# the loss and metric values we will ignore the new class by setting ignore_index=13.

# Not used in code.
watershed_to_class_v2 = {
    "#505050":  0, # Black Background
    "#111111":  1, # Abdominal Wall
    "#212121":  2, # Liver
    "#131313":  3, # Gastrointestinal Tract
    "#121212":  4, # Fat
    "#313131":  5, # Grasper
    "#232323":  6, # Connective Tissue
    "#242424":  7, # Blood
    "#252525":  8, # Cystic Duct
    "#323232":  9, # L-hook Electrocautery
    "#222222": 10, # Gallbladder
    "#333333": 11, # Hepatic Vein
    "#050505": 12, # Liver Ligament
 "#255255255": 13,
    "#000000": 13,
}

# Dictionary mapping watershed_mask colors to their corresponding classes.
# The mapping is 1 x RGB_color_value -> torch.FloatTensor of 3 x class_number

# This version is used in the code.
watershed_to_class_v3 = {
    "50": torch.FloatTensor([ 0,  0,  0]),
    "11": torch.FloatTensor([ 1,  1,  1]),
    "21": torch.FloatTensor([ 2,  2,  2]),
    "13": torch.FloatTensor([ 3,  3,  3]),
    "12": torch.FloatTensor([ 4,  4,  4]),
    "31": torch.FloatTensor([ 5,  5,  5]),
    "23": torch.FloatTensor([ 6,  6,  6]),
    "24": torch.FloatTensor([ 7,  7,  7]),
    "25": torch.FloatTensor([ 8,  8,  8]),
    "32": torch.FloatTensor([ 9,  9,  9]),
    "22": torch.FloatTensor([10, 10, 10]),
    "33": torch.FloatTensor([11, 11, 11]),
    "05": torch.FloatTensor([12, 12, 12]),
   "255": torch.FloatTensor([13, 13, 13]),
    "00": torch.FloatTensor([13, 13, 13]),
}

# Dictionary mapping classes to their corresponding color in color_masks.

# This version is used in the code to create visualization images out of the predictions.
class_to_color = {
     '0': torch.FloatTensor([127, 127, 127]) / 255.,
     '1': torch.FloatTensor([210, 140, 140]) / 255.,
     '2': torch.FloatTensor([255, 114, 114]) / 255.,
     '3': torch.FloatTensor([231,  70, 156]) / 255.,
     '4': torch.FloatTensor([186, 183,  75]) / 255.,
     '5': torch.FloatTensor([170, 255,   0]) / 255.,
     '6': torch.FloatTensor([255,  85,   0]) / 255.,
     '7': torch.FloatTensor([255,   0,   0]) / 255.,
     '8': torch.FloatTensor([255, 255,   0]) / 255.,
     '9': torch.FloatTensor([169, 255, 184]) / 255.,
    '10': torch.FloatTensor([255, 160, 165]) / 255.,
    '11': torch.FloatTensor([  0,  50, 128]) / 255.,
    '12': torch.FloatTensor([111,  74,   0]) / 255.,
    '13': torch.FloatTensor([255, 255, 255]) / 255.,
}

# Dictionary mapping watershed_mask colors to their corresponding color in color_masks.

# Not used in code.
watershed_to_color = {
    '#505050': torch.FloatTensor([127, 127, 127]) / 255.,
    '#111111': torch.FloatTensor([210, 140, 140]) / 255.,
    '#212121': torch.FloatTensor([255, 114, 114]) / 255.,
    '#131313': torch.FloatTensor([231,  70, 156]) / 255.,
    '#121212': torch.FloatTensor([186, 183,  75]) / 255.,
    '#313131': torch.FloatTensor([170, 255,   0]) / 255.,
    '#232323': torch.FloatTensor([255,  85,   0]) / 255.,
    '#242424': torch.FloatTensor([255,   0,   0]) / 255.,
    '#252525': torch.FloatTensor([255, 255,   0]) / 255.,
    '#323232': torch.FloatTensor([169, 255, 184]) / 255.,
    '#222222': torch.FloatTensor([255, 160, 165]) / 255.,
    '#333333': torch.FloatTensor([  0,  50, 128]) / 255.,
    '#050505': torch.FloatTensor([111,  74,   0]) / 255.,
 '#255255255': torch.FloatTensor([255, 255, 255]) / 255.,
    '#000000': torch.FloatTensor([255, 255, 255]) / 255.,
}
