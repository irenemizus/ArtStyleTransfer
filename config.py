simultaneous_tasks_count = 2 # 1 if the levels_num > 3

class Config:
    def __init__(self,
                 content_weight=1e3,
                 style_weight=4e5,
                 tv_weight=1e2,
                 optimizer='lbfgs', model='vgg19', init_method='content+noise',
                 levels_num=2,
                 iters_num=800,
                 noise_factor=0.95,
                 noise_levels=                      (   9,   18,   36,   -1,    0),
                 noise_levels_central_amplitude=    (0.30, 0.20, 0.10, 0.20, 0.20),
                 noise_levels_peripheral_amplitude= (0.20, 0.30, 0.40, 0.10, 0.00),
                 noise_levels_dispersion =          (0.20, 0.30, 0.40, 0.60, 0.30)):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.optimizer = optimizer
        self.model = model
        self.init_method = init_method
        self.levels_num = levels_num # 4 for maximum resolution
        self.iters_num = iters_num # 1500 for maximum quality
        self.noise_factor = noise_factor
        self.noise_levels = noise_levels
        self.noise_levels_central_amplitude = noise_levels_central_amplitude
        self.noise_levels_peripheral_amplitude = noise_levels_peripheral_amplitude
        self.noise_levels_dispersion = noise_levels_dispersion