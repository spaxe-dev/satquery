import os, json, random
random.seed(42)
def make_synthetic_dataset(n, path):
    templates=[("Describe the land-cover","This Sentinel-2 patch shows mixed agricultural fields and sparse settlement."),("Is there water visible?","Yes — a river traverses the southern portion."),("What is the dominant land-cover?","Forest dominates (~60%), with adjacent barren patches."),("Highlight the built-up area","Built-up appears in the northeast quadrant.")]
    data=[]
    for i in range(n):
        q,a=random.choice(templates)
        data.append({"image": f"synthetic_{i}.png","query":q,"answer":a,"split":"train" if i < n*0.9 else "val"})
    os.makedirs(os.path.dirname(path),exist_ok=True)
    import pathlib
    with open(path,"w") as f: json.dump(data,f,indent=2)
    print(f"Synthetic dataset {n} samples -> {path}")
    return data
