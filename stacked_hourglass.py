from torch import nn
import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l.mean() ## l of dim bsize
    

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True, num_landmarks = 72):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim * num_landmarks, out_dim * num_landmarks, kernel_size, stride, padding=(kernel_size-1)//2, bias=True, groups=num_landmarks)
        self.relu = None
        self.bn = None
        self.num_landmarks= num_landmarks
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim * num_landmarks)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim * self.num_landmarks, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, num_landmarks = 72):
        super(Residual, self).__init__()
        
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim * num_landmarks)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False, num_landmarks=num_landmarks)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2) * num_landmarks)
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False, num_landmarks=num_landmarks)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2) * num_landmarks)
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False, num_landmarks=num_landmarks)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False, num_landmarks=num_landmarks)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0, num_landmarks = 72):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f, num_landmarks=num_landmarks)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, nf, num_landmarks=num_landmarks)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn, num_landmarks=num_landmarks)
        else:
            self.low2 = Residual(nf, nf, num_landmarks=num_landmarks)
        self.low3 = Residual(nf, f, num_landmarks=num_landmarks)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):   
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2
    

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim, num_landmarks=72):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False, num_landmarks=num_landmarks)

    def forward(self, x):
        return self.conv(x)
    
class StackedHourglass(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, num_landmarks = 72, hourglass_depth = 4):
        super(StackedHourglass, self).__init__()
        
        self.nstack = nstack
        self.num_landmarks = num_landmarks
        # úvodní síť - jde o převod na dimenzionalitu sítě a případně pooling.
        self.pre = nn.Sequential(
            Conv(3, inp_dim//2, 3, 1, bn=True, relu=True, num_landmarks=num_landmarks),
            Residual(inp_dim//2, inp_dim, num_landmarks=num_landmarks),
            # Pool(2, 2),
            # Residual(128, 128),
            # Residual(128, inp_dim)
        )
        
        # jádro - stacked hourglass
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(hourglass_depth, inp_dim, bn, increase, num_landmarks = num_landmarks),
        ) for i in range(nstack)] )
        
        # odbočující větve pro výpočet loss
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim, num_landmarks = num_landmarks),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True, num_landmarks = num_landmarks)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False, num_landmarks=num_landmarks) for i in range(nstack)] )

        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim, num_landmarks=num_landmarks) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim, num_landmarks=num_landmarks) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = AdaptiveWingLoss()

    def forward(self, x):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)  # resulting shape: (batch, n_stack, num_landmarks, crop_size, crop_size)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = 0
        for i in range(self.nstack):
            combined_loss += self.heatmapLoss(combined_hm_preds[:,i,...], heatmaps)
        # combined_loss = torch.stack(combined_loss, dim=0)
        return combined_loss
    
    


