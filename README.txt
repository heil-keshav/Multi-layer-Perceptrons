Plots are generated by running scripts in MATLAB R2014a.
Dataset-
   Keep all the training data(images) in the folder named steering which is in the same directory as of the script.
Make sure that data.txt file is there.(sorted in alphabetical order to make sure that steering angle corresponds to its correct image file)
Now run the given script in Matlab.
This will generate the figure which shows a graph between training/validation error against the epochs. You can also view the error values on your command prompt screen of matlab.
To change learning parameters in task 2, open the script and go to line 49 for changing the no. of epochs, go to line 52 for changing the learning rate and go to line 104 for changing the Minibatch size. For changing the dropout probability go to line 129 in l32drop.m script
