def setup(GPU):
    from Dir_learning.P4_transferLearning.P1_Returns.src.P1_revisited import run_p1r
    import torch
    if GPU == 0:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        import Dir_learning.P4_transferLearning.P1_Returns.Settings.Settings0 as SS
    elif GPU == 1:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        import Dir_learning.P4_transferLearning.P1_Returns.Settings.Settings1 as SS

    print(f"SETUP    {GPU}     {device}     {SS.model_name}")

    count = 0
    while count <= len(SS.seeds):
        run_p1r(GPU)
