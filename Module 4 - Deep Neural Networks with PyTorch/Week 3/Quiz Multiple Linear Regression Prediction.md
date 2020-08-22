## Multiple Linear Regression Prediction
> 
> Total points 3
> 
> 1.
> 
> Question 1
> 
> Including the bias how many parameters does the following object have
>  
> model=nn.Linear(6,1)
> 

    7
> 
> 1 point
> 
> Enter answer here
> 
> 2.
> 
> Question 2
> 
> How would you create a linear object with ten input features?
> 
> 1 point
> 

      model=nn.Linear(10,1) 
> 
>  model=nn.Linear(1,10) 
> 
> 3.
> 
> Question 3
> 
> How do you calculate the gradient and perform the update for Multiple Linear Regression with 10 input variables?
> 
> 1 point
> 
> 
> loss.backward(d=10)
> 
> optimizer.step(d=10)
> 
> loss.backward()
> 
> optimizer.step(d=10)
> 
>

     loss.backward()
     optimizer.step()
> 
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/PC4pN/multiple-linear-regression-prediction/attempt#Tunnel Vision Close
