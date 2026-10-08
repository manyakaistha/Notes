import re

with open('week-1-notes.md', 'r') as f:
    text = f.read()

# Make sure the equation is the original \xLongleftrightarrow or \stackrel
# The user originally had \xLongleftrightarrow{\quad\mathcal{F}\quad}
# I will make it \xleftrightarrow{\mathcal{F}} since KaTeX and MathJax both support it cleanly without missing macro issues if configured right. Or better, just standard LaTeX \stackrel{\mathcal{F}}{\longleftrightarrow} which I already have.
# Let's verify it is \stackrel{\mathcal{F}}{\longleftrightarrow}

if r'\stackrel{\mathcal{F}}{\longleftrightarrow}' in text:
    print("Equation is currently: \stackrel{\mathcal{F}}{\longleftrightarrow}")
elif r'\xLongleftrightarrow' in text:
    print("Equation is currently: \xLongleftrightarrow")
elif r'\xleftrightarrow' in text:
    print("Equation is currently: \xleftrightarrow")
else:
    print("Equation not found")

