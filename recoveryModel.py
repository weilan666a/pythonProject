import paddle
def load_model_optimizer(checkpoint_path, model=None,device="xpu"):
    checkpoint = paddle.load(checkpoint_path,map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['state_dict'])

    return checkpoint['epoch']