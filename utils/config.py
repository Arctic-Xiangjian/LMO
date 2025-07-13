def save_config(config_path, open_type, time, args):
    with open(config_path, open_type) as f_obj:
        f_obj.write('----------------' + time + '----------------' + '\n\n')
        for arg in vars(args):
            f_obj.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f_obj.write('\n===================================================')
        f_obj.write('\n\n')
