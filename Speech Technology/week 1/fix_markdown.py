import re

with open('week-1-notes.md', 'r') as f:
    text = f.read()

# Fix the yaml frontmatter links
text = text.replace('"[[speech-chain]]"', '"[speech-chain](../notes/concepts/speech-chain.md)"')
text = text.replace('"[[nyquist-sampling-theorem]]"', '"[nyquist-sampling-theorem](../notes/concepts/nyquist-sampling-theorem.md)"')
text = text.replace('"[[linear-time-invariant-systems]]"', '"[linear-time-invariant-systems](../notes/concepts/linear-time-invariant-systems.md)"')
text = text.replace('"[[discrete-fourier-transform]]"', '"[discrete-fourier-transform](../notes/concepts/discrete-fourier-transform.md)"')
text = text.replace('"[[rational-filters-poles-zeros]]"', '"[rational-filters-poles-zeros](../notes/concepts/rational-filters-poles-zeros.md)"')
text = text.replace('"[[autocorrelation-pitch-estimation]]"', '"[autocorrelation-pitch-estimation](../notes/concepts/autocorrelation-pitch-estimation.md)"')
text = text.replace('"[[short-time-fourier-transform]]"', '"[short-time-fourier-transform](../notes/concepts/short-time-fourier-transform.md)"')
text = text.replace('"[[source-filter-model]]"', '"[source-filter-model](../notes/concepts/source-filter-model.md)"')
text = text.replace('"[[mel-scale-filter-banks]]"', '"[mel-scale-filter-banks](../notes/concepts/mel-scale-filter-banks.md)"')
text = text.replace('"[[cepstral-analysis-mfcc]]"', '"[cepstral-analysis-mfcc](../notes/concepts/cepstral-analysis-mfcc.md)"')

text = text.replace('prev: "[[Knowledge-Map|Knowledge Map]]"', 'prev: "[Knowledge Map](../Knowledge-Map.md)"')
text = text.replace('next: "[[week 2/week-2-notes|Week 2 Notes]]"', 'next: "[Week 2 Notes](../week 2/week-2-notes.md)"')
text = text.replace('map: "[[Knowledge-Map]]"', 'map: "[Knowledge-Map](../Knowledge-Map.md)"')

# Change \stackrel back to better formatting
text = text.replace(r'\stackrel{\mathcal{F}}{\longleftrightarrow}', r'\xleftrightarrow{\mathcal{F}}')

with open('week-1-notes.md', 'w') as f:
    f.write(text)

