### 3.3 Optimization in PyTorch
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> What does the following line of code do?
> 
> _optimizer.step()_
> 
> 1 / 1 point
> 

      Makes an update to its parameters 
> 
>  Makes a prediction 
> 
>  Clears the gradient 
> 
>  Computes the gradient of the loss with respect to all the learnable parameters 
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
> What's wrong with the following lines of code?
> 
> 
> 
> optimizer = optim.SGD(model.parameters(), lr = 0.01)
> 
> model=linear_regression(1,1)
> 
> 
> 1 / 1 point
> 

      The model object has not been created. As such, the argument that specifies what Tensors should be optimized does not exist 
> 
>  There is no loss function 
> 
>  You have to clear the gradient 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/pIFEV/3-3-optimization-in-pytorch/attempt?redirectToCover=true#Tunnel Vision Close
