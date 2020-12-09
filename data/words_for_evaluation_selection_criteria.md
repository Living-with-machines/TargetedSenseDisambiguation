The criteria used to select the words for evaluation are based on a mix of subjective and objective criteria:

**Subjective** – the rationale:

* words whose use and definitions are a bit blurred (nation, democracy, art, power)
* words with a in interesting historical relevance (nation, democracy, slave, power, labour)
* words with possible different meanings/contexts (labour: work force Vs party)
* words that could be interesting should we do some future work on sentiment analysis (happiness, anger)
* words whose results could (or could not) be interesting (woman, man, slave)
* a random word, completely out of context (apple)

**Objective** – the rationale:
* words that have multiple senses
* words that have multiple senses in the period we are interested in (1750-1920)
* number of quotations associated with the multiple senses
* number of quotations associated with the senses in use in the period we are interested in (1750-1920)

The selection:
* initial subjective selection, followed by
* quantitative analysis of data available (https://github.com/Living-with-machines/PlaceLinking/blob/parse-quick/quick/parse_quick_gt.ipynb)

Initial selection:  
`lemmas = ['nation', 'art', 'technology', 'labour', 'power', 'democracy', 'woman', 'man', 'slave', 'apple', 'anger', 'happiness']`

Analysis:

```
NATION
Number of definitions :  4

nation_nn01
Number of senses:  18
Number of senses (1750-1920):  12
Number of total quotes:  129
Number of total quotes (1750-1920):  102

nation_jj01
Number of senses:  1
Number of senses (1750-1920):  1
Number of total quotes:  6
Number of total quotes (1750-1920):  6

nation_rb01
Number of senses:  1
Number of senses (1750-1920):  1
Number of total quotes:  9
Number of total quotes (1750-1920):  9

nation_nn02
Number of senses:  2
Number of senses (1750-1920):  2
Number of total quotes:  12
Number of total quotes (1750-1920):  12


ART
Number of definitions :  4

art_nn01
Number of senses:  20
Number of senses (1750-1920):  17
Number of total quotes:  227
Number of total quotes (1750-1920):  213

art_vb01
Number of senses:  4
Number of senses (1750-1920):  0
Number of total quotes:  13
Number of total quotes (1750-1920):  0

art_nn02
Number of senses:  1
Number of senses (1750-1920):  1
Number of total quotes:  6
Number of total quotes (1750-1920):  6

art_vb02
Number of senses:  5
Number of senses (1750-1920):  0
Number of total quotes:  10
Number of total quotes (1750-1920):  0


TECHNOLOGY
Number of definitions :  1

technology_nn01
Number of senses:  7
Number of senses (1750-1920):  6
Number of total quotes:  35
Number of total quotes (1750-1920):  34


LABOUR
Number of definitions :  2

labour_nn01
Number of senses:  18
Number of senses (1750-1920):  14
Number of total quotes:  159
Number of total quotes (1750-1920):  137

labour_vb01
Number of senses:  26
Number of senses (1750-1920):  19
Number of total quotes:  271
Number of total quotes (1750-1920):  227


POWER
Number of definitions :  3

power_nn01
Number of senses:  38
Number of senses (1750-1920):  33
Number of total quotes:  331
Number of total quotes (1750-1920):  311

power_vb01
Number of senses:  4
Number of senses (1750-1920):  2
Number of total quotes:  19
Number of total quotes (1750-1920):  14

power_nn02
Number of senses:  1
Number of senses (1750-1920):  1
Number of total quotes:  6
Number of total quotes (1750-1920):  6


POWER
Number of definitions :  3

power_nn01
Number of senses:  38
Number of senses (1750-1920):  33
Number of total quotes:  331
Number of total quotes (1750-1920):  311

power_vb01
Number of senses:  4
Number of senses (1750-1920):  2
Number of total quotes:  19
Number of total quotes (1750-1920):  14

power_nn02
Number of senses:  1
Number of senses (1750-1920):  1
Number of total quotes:  6
Number of total quotes (1750-1920):  6


DEMOCRACY
Number of definitions :  1

democracy_nn01
Number of senses:  7
Number of senses (1750-1920):  7
Number of total quotes:  60
Number of total quotes (1750-1920):  60


WOMAN
Number of definitions :  2

woman_nn01
Number of senses:  17
Number of senses (1750-1920):  16
Number of total quotes:  212
Number of total quotes (1750-1920):  211

woman_vb01
Number of senses:  5
Number of senses (1750-1920):  5
Number of total quotes:  27
Number of total quotes (1750-1920):  27


MAN
Number of definitions :  7

man_nn01
Number of senses:  73
Number of senses (1750-1920):  62
Number of total quotes:  604
Number of total quotes (1750-1920):  555

man_pr01
Number of senses:  1
Number of senses (1750-1920):  0
Number of total quotes:  13
Number of total quotes (1750-1920):  0

man_nn02
Number of senses:  1
Number of senses (1750-1920):  0
Number of total quotes:  8
Number of total quotes (1750-1920):  0

man_jj01
Number of senses:  1
Number of senses (1750-1920):  0
Number of total quotes:  3
Number of total quotes (1750-1920):  0

man_nn03
Number of senses:  1
Number of senses (1750-1920):  0
Number of total quotes:  6
Number of total quotes (1750-1920):  0

man_vb01
Number of senses:  16
Number of senses (1750-1920):  11
Number of total quotes:  105
Number of total quotes (1750-1920):  88

man_nn04
Number of senses:  3
Number of senses (1750-1920):  0
Number of total quotes:  8
Number of total quotes (1750-1920):  0


SLAVE
Number of definitions :  4

slave_nn01
Number of senses:  11
Number of senses (1750-1920):  8
Number of total quotes:  65
Number of total quotes (1750-1920):  56

slave_vb02
Number of senses:  1
Number of senses (1750-1920):  0
Number of total quotes:  2
Number of total quotes (1750-1920):  0

slave_vb01
Number of senses:  12
Number of senses (1750-1920):  11
Number of total quotes:  33
Number of total quotes (1750-1920):  30

slave_nn02
Number of senses:  1
Number of senses (1750-1920):  1
Number of total quotes:  14
Number of total quotes (1750-1920):  14


APPLE
Number of definitions :  2

apple_nn01
Number of senses:  20
Number of senses (1750-1920):  14
Number of total quotes:  172
Number of total quotes (1750-1920):  143

apple_vb01
Number of senses:  2
Number of senses (1750-1920):  2
Number of total quotes:  12
Number of total quotes (1750-1920):  12


ANGER
Number of definitions :  2

anger_vb01
Number of senses:  7
Number of senses (1750-1920):  5
Number of total quotes:  55
Number of total quotes (1750-1920):  48

anger_nn01
Number of senses:  6
Number of senses (1750-1920):  3
Number of total quotes:  43
Number of total quotes (1750-1920):  27


HAPPINESS
Number of definitions :  1

happiness_nn01
Number of senses:  5
Number of senses (1750-1920):  5
Number of total quotes:  42
Number of total quotes (1750-1920):  42

```
