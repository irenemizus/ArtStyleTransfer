simultaneous_tasks_count = 2 # 1 if the levels_num > 2

class Config:
    """ A class with configuration settings """
    def __init__(self,
                 content_weight=1e3,                                                        # weight factor for content loss, type=float
                 style_weight=4e5,                                                          # weight factor for style loss, type=float
                 tv_weight=1e2,                                                             # weight factor for total variation loss, type=float
                 optimizer='lbfgs',                                                         # choices=['lbfgs', 'adam'], type=str
                 model='vgg19',                                                             # choices=['vgg19'], type=str
                 init_method='content+noise',                                               # choices=['random', 'content+noise', 'style'], type=str
                 levels_num=2,                                                              # number of pyramid levels, type=int, = 4 for maximum resolution
                 iters_num=800,                                                             # number of iterations to perform, type=int, = 1500 for maximum quality
                 noise_factor=0.95, #0.5,                                                         # strength of noise, which is added to the initial image, type=float
                 noise_levels=                      (   9,   18,   36,   -1,    0),         # tuple of noise spots number/size along the shortest axis of the content image for a few noise levels, type=tuple
                 noise_levels_central_amplitude=    (0.30, 0.20, 0.10, 0.20, 0.20),         # tuple of noise gaussian envelope strength in the center of the image for a few noise levels, type=tuple
                 noise_levels_peripheral_amplitude= (0.20, 0.30, 0.40, 0.10, 0.00),         # tuple of noise gaussian envelope strength at the periphery of the image for a few noise levels, type=tuple
                 noise_levels_dispersion =          (0.20, 0.30, 0.40, 0.60, 0.30)):        # tuple of noise gaussian envelope dispersion for a few noise levels, type=tuple
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.optimizer = optimizer
        self.model = model
        self.init_method = init_method
        self.levels_num = levels_num
        self.iters_num = iters_num
        self.noise_factor = noise_factor
        self.noise_levels = noise_levels
        self.noise_levels_central_amplitude = noise_levels_central_amplitude
        self.noise_levels_peripheral_amplitude = noise_levels_peripheral_amplitude
        self.noise_levels_dispersion = noise_levels_dispersion