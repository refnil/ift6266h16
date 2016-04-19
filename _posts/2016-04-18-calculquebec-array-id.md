---
title: Using array task (or rather not using it)
---

So I wanted to try many combination of hyperparameters automatically. Doing my computation on the Calcul Québec, one way to do it would be to lunch multiple task invidually at the same time, which could get annoying to manage. 

Another way would be to use the task array feature of the cluster. An example can be found [here](https://wiki.calculquebec.ca/w/Example_scripts_for_multiple_serial_simulations,_OpenMP_jobs,_and_hybrid_jobs#Lot_de_t.C3.A2ches_s.C3.A9quentielles) under the sequential task array section. Such script will be executed as many times as you want and then you can use the environnement variable $PBS_ARRAYID to decide which hyperparameters to use to train the neural net.

Unfortunatly, for some reason, it didn't work with CUDA and I couldn't figure why.

```
/usr/bin/ld: cannot find -lcuda_ndarray
collect2: ld returned 1 exit status
Traceback (most recent call last):
    File "train_many.py", line 1, in <module>
        import train
    File "/home2/ift6ed10/ift6266h16/projects/cat_dog/train.py", line 8, in <module>
        from theano import tensor
    File "/home2/ift6ed10/anaconda3/lib/python3.5/site-packages/theano/__init__.py", line 96, in <module>
        if hasattr(theano.tests, "TheanoNoseTester"):
AttributeError: module 'theano' has no attribute 'tests'
```

The solution I used  to circumvent the problem was to make a script which create standard tasks dynamically.
