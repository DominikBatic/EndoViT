import torch
import torchvision.transforms as transforms

class HighPassFilter():
    def __init__(self, radius, imgShape):
        self.radius = radius
        self.imgShape = imgShape
        
        # construct the Gaussian filter
        self.filter = self.gaussianHP(radius, imgShape)
        
        # transformation to greyscale
        self.transform = transforms.Grayscale()
 
    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
  
    def gaussianHP(self, D0, imgShape):
        rows, cols = imgShape
        center = ((rows-1)/2, (cols-1)/2)
        
        # creating a grid of x and y values
        position = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
        
        # creating the same grid but with x-center and y-center at every value
        center_grid = torch.concatenate(
                        [
                            torch.unsqueeze(torch.full((rows, cols), center[0], dtype=torch.float32), dim=0), 
                            torch.unsqueeze(torch.full((rows, cols), center[1], dtype=torch.float32), dim=0)
                        ], 
                        dim=0)
        
        # build the filter
        g_filter = 1 - torch.exp((-self.squared_distance(position, center_grid)) / (2*D0**2))
            
        return g_filter

    def apply(self, pred, target):
        """
        Inputs:
            pred: 
                torch.tensor (B, C, H, W) - MAE reconstruction of the input
            target: 
                torch.tensor (B, C, H, W) - Ground Truth targets
    
        Output:
            hp_pred:
                torch.tensor (B, 1, H, W) - High Pass Filtered predictions
            hp_target
                torch.tensor (B, 1, H, W) - High Pass Filtered targets
            applied_filter:
                torch.tensor (H, W) - Filter which was applied
        """
        device = pred.device

        # make the images grayscale
        pred = self.transform(pred).to(torch.complex64)
        target = self.transform(target).to(torch.complex64)

        # apply fft
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # shift frequencies such that low frequencies are in the center of the image
        pred_centered = torch.fft.fftshift(pred_fft)
        target_centered = torch.fft.fftshift(target_fft)

        # apply the filters
        hp_pred = pred_centered * self.filter.to(device)
        hp_target = target_centered * self.filter.to(device)

        # reverse the shift
        hp_pred = torch.fft.ifftshift(hp_pred)
        hp_target = torch.fft.ifftshift(hp_target)

        # apply ifft
        hp_pred = torch.fft.ifft2(hp_pred)
        hp_target = torch.fft.ifft2(hp_target)
        
        # bring the results back to torch.tensor
        hp_pred = torch.abs(hp_pred)
        hp_target = torch.abs(hp_target)
        
        return (hp_pred, hp_target, self.filter)