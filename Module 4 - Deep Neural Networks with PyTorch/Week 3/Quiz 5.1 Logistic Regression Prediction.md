## 5.1 Logistic Regression: Prediction
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> What line of code is equivalent to:
> 
> model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
> 
> 
> 1 / 1 point

     
     class logistic_regression(nn.Module):
     
         # Constructor
     
        def __init__(self, n_inputs):
     
             super(logistic_regression, self).__init__()
     
             self.linear = nn.Linear(n_inputs, 1)
     
         # Prediction
     
         def forward(self, x):
     
             yhat = torch.sigmoid(self.linear(x))
     
             return yhat
     
     model = logistic_regression(1)
> 
>
> 
>  yhat = torch.sigmoid(self.linear(x)) 
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
> How would you apply the sigmoid function to the tensor z
>
> z=torch.arange(-100,100,0.1).view(-1, 1)
> 
> 1 / 1 point
> 
     sig=nn. Sigmoid ()
     
     yhat=sig(z)
> 
> 
> Check
> 
> Correct
> 
> correct
>  

     yhat= torch.sigmoid(z)
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/hLnhH/5-1-logistic-regression-prediction/attempt?redirectToCover=true#Tunnel Vision Close
