import numpy as np
# A tiny corpus: <s> I like tea. P(sentence)=product of conditionals.
conditionals=[.4,.6,.5]; bigram=[.4,.3,.2]
for label,p in (("full history",conditionals),("bigram",bigram)):
    print(f"{label}: P={np.prod(p):.6f}, token NLL={-np.log(p).round(4)}")
