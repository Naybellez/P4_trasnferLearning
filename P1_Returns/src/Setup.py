def setup(GPU):
    from src.P1_revisited import run_p1r
    import torch
    
    if GPU == 0:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        import Settings.Settings0 as SS
    elif GPU == 1:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        import Settings.Settings1 as SS

    print(f"SETUP    {GPU}     {device}     {SS.model_name}")

    count = 0
    print("set up len seeds: ",len(SS.seeds))
    while count <= len(SS.seeds):
        run_p1r(GPU)
        print("ran")
        count +=1
