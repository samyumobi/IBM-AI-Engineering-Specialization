### Prediction in One Dimension
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> what is wrong with the following lines of code:
> 
> class LR():
> 
>     # Constructor
> 
>     def __init__(self, input_size, output_size):
> 
>         # Inherit from parent
> 
>         super(LR, self).__init__()
> 
>         self.linear = nn.Linear(input_size, output_size)
> 
>     # Prediction function
> 
>     def forward(self, x):
> 
>         out = self.linear(x)
> 
>         return out
> 
> 
> 1 / 1 point
> 

      its missing **nn.Module** 
> 
>  there is no call function 
> 
> Check
> 
> Correct
> 
> correct
> 
> 2.
> 
> Question 2
> 
> Consider the following lines of code. How many Parameters does the object model have?
> 
> 
> from torch.nn import Linear
> 
> model=Linear(in_features=1,out_features=1)
> 
> 
> 1 / 1 point
> 
>  1 
> 

      2 
> 
>  3 
> 
>  None of the above 
> 
> Check
> 
> Correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/3t0Vs/prediction-in-one-dimension/attempt?redirectToCover=true#Tunnel Vision Close
