## Multiple Output Linear Regression
> 
> Total points 2
> 
> 1.
> 
> Question 1
> 
> What is true about the following lines of code?
> 
> class linear_regression(nn.Module):
> 
>     def __init__(self,input_size,output_size):
> 
>         super(linear_regression,self).__init__()
> 
>         self.linear=nn.Linear(input_size,output_size)
> 
>     def forward(self,x):
> 
>         yhat=self.linear(x)
> 
>         return yhat
> 
> model=linear_regression(3,10)  
>  
> 1 point
> 
>  The output of the model will have 10 rows 
> 

      The output of the model will have 10 columns 
> 
> 2.
> 
> Question 2
> 
> What parameters do you have to change to the method backwards() when you train Multiple Output Linear Regression compared to regular Linear Regression?
> 
> 1 point
> 

      None of them 
> 
>  You have to specify the number of the output variables 
> 
>  All of them
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/TH6iH/multiple-output-linear-regression/attempt#Tunnel Vision Close
