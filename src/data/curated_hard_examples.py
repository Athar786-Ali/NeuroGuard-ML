"""
A small, hand-curated set of sentiment examples that are grammatically
natural (unlike the synthetic noise from src/features/noise.py, which
often produces broken phrases like "the not closest thing") but
semantically hard -- the kind of phrasing that trips up sentiment
classifiers even though a human reads them correctly without effort.

Why this exists: analysis of the original 5,000-sample dataset showed
its "failure" examples were almost all synthetic negation-insertion
artifacts (e.g. "the not closest thing to the experience"), not
natural language. The model had very little signal for real semantic
traps like negation, sarcasm-via-negative-words, or intensifiers that
flip a negative word positive ("disgustingly good"). This set targets
exactly those gaps.

Each entry is (text, true_label) where true_label follows the same
convention as SST-2: 1 = positive, 0 = negative. Labels reflect the
correct human judgment of the sentence, not any model's prediction.

This is a few hundred examples, not thousands -- it's meant to be
blended into the larger SST-2-sampled dataset as a targeted booster,
not to replace real-world data volume.
"""

CURATED_HARD_EXAMPLES = [
    # ---------------- Negation flips ----------------
    ("not the worst movie I've ever seen", 1),
    ("not the best film, but far from the worst", 1),
    ("I wouldn't say it's bad", 1),
    ("this isn't terrible", 1),
    ("hardly a disaster", 1),
    ("never boring, not once", 1),
    ("not without its charms", 1),
    ("can't say I disliked it", 1),
    ("not a bad way to spend an evening", 1),
    ("doesn't fail to entertain", 1),
    ("not unimpressive for a debut feature", 1),
    ("never dull, even for a second", 1),
    ("I don't regret watching this one bit", 1),
    ("this movie isn't without flaws, but it works", 1),
    ("not the masterpiece some claimed, but solid", 1),
    ("not great, not terrible either", 1),
    ("the plot wasn't predictable at all", 1),
    ("acting that doesn't disappoint", 1),
    ("not a single dull moment", 1),
    ("it never stops surprising you", 1),

    ("not the triumph the trailers promised", 0),
    ("not as good as the original", 0),
    ("doesn't live up to the hype", 0),
    ("never quite finds its footing", 0),
    ("not worth the ticket price", 0),
    ("can't recommend this one", 0),
    ("not the director's finest work", 0),
    ("doesn't justify its runtime", 0),
    ("never manages to be funny", 0),
    ("not a film I'd watch twice", 0),
    ("hardly the comeback fans hoped for", 0),
    ("doesn't hold together as a story", 0),
    ("not nearly as clever as it thinks it is", 0),
    ("never escapes its own clichés", 0),
    ("not worth sitting through twice", 0),

    # ---------------- Negative-word-as-praise (sarcasm-adjacent intensifiers) ----------------
    ("disgustingly good", 1),
    ("disgustingly addictive", 1),
    ("stupidly entertaining", 1),
    ("ridiculously fun", 1),
    ("absurdly enjoyable", 1),
    ("annoyingly catchy", 1),
    ("criminally underrated", 1),
    ("sinfully delicious", 1),
    ("painfully good", 1),
    ("dangerously addictive", 1),
    ("insanely well made", 1),
    ("frighteningly good performances", 1),
    ("obscenely talented cast", 1),
    ("disturbingly brilliant", 1),
    ("savagely funny", 1),
    ("brutally honest and all the better for it", 1),
    ("wickedly clever plot twists", 1),
    ("scandalously underappreciated film", 1),
    ("recklessly entertaining from start to finish", 1),
    ("shamelessly fun popcorn movie", 1),

    ("genuinely disappointing", 0),
    ("truly forgettable", 0),
    ("painfully boring", 0),
    ("disappointingly flat", 0),
    ("frustratingly dull", 0),
    ("hopelessly mediocre", 0),
    ("tediously long", 0),
    ("genuinely unpleasant to sit through", 0),
    ("depressingly predictable", 0),
    ("astonishingly bad writing", 0),

    # ---------------- Comparative / mixed sentiment ----------------
    ("sharper, not duller, than the original", 1),
    ("less polished but more heartfelt than the sequel", 1),
    ("rougher around the edges, but more honest", 1),
    ("a slow start that pays off by the end", 1),
    ("messy in places but ultimately rewarding", 1),
    ("uneven, yet the highs are worth it", 1),
    ("imperfect, but I loved every minute", 1),
    ("flawed in execution, brilliant in ambition", 1),
    ("a rough draft of a great idea, still worth watching", 1),
    ("clunky dialogue, but the performances carry it", 1),

    ("polished but utterly soulless", 0),
    ("technically impressive, emotionally empty", 0),
    ("a great cast wasted on a weak script", 0),
    ("gorgeous visuals can't save a hollow story", 0),
    ("starts strong, collapses by the third act", 0),
    ("ambitious but ultimately a mess", 0),
    ("polished on the surface, hollow underneath", 0),
    ("good intentions, poor execution", 0),
    ("looks great, says nothing", 0),
    ("competent but forgettable", 0),

    # ---------------- Faint / understated praise and criticism ----------------
    ("it was fine, I guess", 1),
    ("decent enough way to kill two hours", 1),
    ("better than I expected, honestly", 1),
    ("surprisingly watchable", 1),
    ("held my attention more than I thought it would", 1),
    ("a pleasant surprise", 1),
    ("quietly impressive", 1),
    ("modest but satisfying", 1),
    ("nothing groundbreaking, but enjoyable", 1),
    ("exceeded my admittedly low expectations", 1),

    ("just okay, nothing more", 0),
    ("watchable, but that's the best I can say", 0),
    ("fine if you have nothing else to watch", 0),
    ("underwhelming, if I'm honest", 0),
    ("didn't really go anywhere", 0),
    ("left me feeling pretty indifferent", 0),
    ("not memorable in the slightest", 0),
    ("a forgettable couple of hours", 0),
    ("fell short of even modest expectations", 0),
    ("didn't quite land for me", 0),

    # ---------------- Rhetorical questions / indirect sentiment ----------------
    ("how is this movie not more popular?", 1),
    ("why did nobody tell me this was this good?", 1),
    ("who knew a sequel could be this much fun?", 1),
    ("could the ending have been any more perfect?", 1),
    ("is there a more underrated film this year?", 1),

    ("how did this get greenlit?", 0),
    ("why does this movie exist?", 0),
    ("who thought this script was ready?", 0),
    ("could the pacing be any slower?", 0),
    ("is there a more wasted premise than this?", 0),

    # ---------------- Long, naturally-phrased mixed examples ----------------
    ("I went in expecting to hate it and came out genuinely impressed", 1),
    ("everyone told me to skip this and I'm glad I didn't listen", 1),
    ("it shouldn't work on paper, but somehow it does", 1),
    ("I can't believe how much I ended up enjoying this", 1),
    ("this had no right to be as good as it was", 1),
    ("I wanted to like this more than I actually did", 0),
    ("everyone raved about this and I just don't see it", 0),
    ("it had every ingredient for greatness and still missed", 0),
    ("I really tried to enjoy this, but couldn't", 0),
    ("this had no right to be this disappointing given the talent involved", 0),
]
