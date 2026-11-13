import numpy as np
p = {("<s>", "I"): .4, ("I", "like"): .6, ("like", "tea"): .5}
sentence = ["<s>", "I", "like", "tea"]
full = [p[(sentence[i-1], sentence[i])] for i in range(1, len(sentence))]
bigram = [.4, .3, .2]
for name, factors in (("full-history toy conditionals", full), ("bigram approximation", bigram)):
    joint = float(np.prod(factors))
    print(f"{name}: factors={factors}, P={joint:.6f}, logP={np.log(joint):.6f}")
