content_weight = 1e3
style_weight = 4e5 #1e5
tv_weight = 1e3
optimizer = 'lbfgs'
model = 'vgg19'
init_method = 'content'
levels_num = 2 # 4 for maximum resolution
iters_num = 800 # 1500 for maximum quality

noise_factor = 0.8 #0.9

simultaneous_tasks_count = 2  # 1 if the levels_num > 3