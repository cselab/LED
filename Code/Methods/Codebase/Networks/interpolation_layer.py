#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import torch.nn.functional as nnf
import torch.nn as nn

# x = torch.rand(5, 1, 44, 44)
# out = nnf.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)


class interpolationLayer(nn.Module):
    def __init__(self, size):
        super(interpolationLayer, self).__init__()
        self.size = size

    def forward(self, x):
        # print("# interpolationLayer #")
        # print(x.size())
        x = nnf.interpolate(x,
                            size=self.size,
                            mode='bicubic',
                            align_corners=False)
        # print(x.size())
        return x

    def sendModelToCuda(self):
        pass
