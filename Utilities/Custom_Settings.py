def apply_resnet_settings(arg):
    arg.batch_size = 32 # orig paper trained all networks with batch_size=128
    arg.no_epochs = 200
    arg.data_augmentation = True
    arg.lr_schedule = True
    return arg