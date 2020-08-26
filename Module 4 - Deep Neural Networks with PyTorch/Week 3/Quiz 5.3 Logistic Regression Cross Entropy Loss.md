## 5.3 Logistic Regression Cross Entropy Loss
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> What cost function should used for logistic regression?
> 
> 1 / 1 point
> 

      Cross Entropy 
> 
>  Mean Squared Error 
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
> What function does the following lines of code perform:
> 
> 
> def criterion(yhat,y):   
> 
>    out = - 1 * torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))  
> 
>   return out
> 
> 
> 1 / 1 point
> 

     calculate the Cross Entropy loss or cost. 
> 
>  calculate the mean squared error 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/b7n47/5-3-logistic-regression-cross-entropy-loss/attempt?redirectToCover=true#Tunnel Vision Close
