class Config:
    def __init__(self, content_weight=1e3, style_weight=4e5, tv_weight=1e3, optimizer='lbfgs', model='vgg19', init_method='content', levels_num=2, iters_num=800, noise_factor=0.95, simultaneous_tasks_count=2):
        self.content_weight = content_weight
        self.style_weight = style_weight #1e5
        self.tv_weight = tv_weight
        self.optimizer = optimizer
        self.model = model
        self.init_method = init_method
        self.levels_num = levels_num # 4 for maximum resolution
        self.iters_num = iters_num # 1500 for maximum quality
        self.noise_factor = noise_factor #0.9
        self.simultaneous_tasks_count = simultaneous_tasks_count  # 1 if the levels_num > 3