for d in (512,256):
    warmup=4000
    print(f"d_model={d}")
    for step in (1,1000,4000,8000):
        lr=d**-0.5*min(step**-0.5,step*warmup**-1.5)
        print(f"  step={step:4d} lr={lr:.10f}")
