import re

with open('week-1-notes.md', 'r') as f:
    text = f.read()

# Revert concepts
text = text.replace('"[speech-chain](../notes/concepts/speech-chain.md)"', '"[[speech-chain]]"')
text = text.replace('"[nyquist-sampling-theorem](../notes/concepts/nyquist-sampling-theorem.md)"', '"[[nyquist-sampling-theorem]]"')
text = text.replace('"[linear-time-invariant-systems](../notes/concepts/linear-time-invariant-systems.md)"', '"[[linear-time-invariant-systems]]"')
text = text.replace('"[discrete-fourier-transform](../notes/concepts/discrete-fourier-transform.md)"', '"[[discrete-fourier-transform]]"')
text = text.replace('"[rational-filters-poles-zeros](../notes/concepts/rational-filters-poles-zeros.md)"', '"[[rational-filters-poles-zeros]]"')
text = text.replace('"[autocorrelation-pitch-estimation](../notes/concepts/autocorrelation-pitch-estimation.md)"', '"[[autocorrelation-pitch-estimation]]"')
text = text.replace('"[short-time-fourier-transform](../notes/concepts/short-time-fourier-transform.md)"', '"[[short-time-fourier-transform]]"')
text = text.replace('"[source-filter-model](../notes/concepts/source-filter-model.md)"', '"[[source-filter-model]]"')
text = text.replace('"[mel-scale-filter-banks](../notes/concepts/mel-scale-filter-banks.md)"', '"[[mel-scale-filter-banks]]"')
text = text.replace('"[cepstral-analysis-mfcc](../notes/concepts/cepstral-analysis-mfcc.md)"', '"[[cepstral-analysis-mfcc]]"')

# Revert frontmatter nav links
text = text.replace('prev: "[Knowledge Map](../Knowledge-Map.md)"', 'prev: "[[Knowledge-Map|Knowledge Map]]"')
text = text.replace('next: "[Week 2 Notes](../week 2/week-2-notes.md)"', 'next: "[[week 2/week-2-notes|Week 2 Notes]]"')
text = text.replace('map: "[Knowledge-Map](../Knowledge-Map.md)"', 'map: "[[Knowledge-Map]]"')

# Revert equation back to original OR the corrected version from step 1?
# In step 1, I fixed `\xLongleftrightarrow{\quad\mathcal{F}\quad}` to `\stackrel{\mathcal{F}}{\longleftrightarrow}`
# Then my rogue script changed it to `\xleftrightarrow{\mathcal{F}}`
# The user's original equation was `\xLongleftrightarrow{\quad\mathcal{F}\quad}` but they asked me to fix it because it doesn't render well.
# I will put it to standard \stackrel{\mathcal{F}}{\longleftrightarrow} maybe?
# Or wait, what if the user PREFERRED `\xLongleftrightarrow` because they have an Obsidian plugin for it?
# The prompt for step 1 was: "Fix this in the notes for week 1 under speech technology: \xLongleftrightarrow..."
# It was broken for them too. So \stackrel is correct. 
# But let's just make sure \stackrel is restored.
text = text.replace(r'\xleftrightarrow{\mathcal{F}}', r'\stackrel{\mathcal{F}}{\longleftrightarrow}')

with open('week-1-notes.md', 'w') as f:
    f.write(text)

