---
title: Cat and Dog (Part 1)
---

So I've decided to work on the cat and dogs project. I do not have result to present yet. I've been familiarising myself with theano, blocks and hades for the last 2 weeks while building some blocks that may or may not be usefull along the way. You can find a more up to date version of my code in the "wip" branch of my github. It contains pretty basic stuff like early stopping, saving the best parameters, etc.

Here is a short list of "goal" that I'll throw out there:

-   Use dropout.

    There is the `apply_dropout` function in `blocks.bricks.conv`, I just need to figure out how to use it properly.

-   Try to explore a lot of hyperparameter with the help of Hades. 

    The option `-t` of PBS to create a job array could proove usefull there.

-   Take the time to make some kind of visualizion of a neural network running, be it a gif, video or program.
