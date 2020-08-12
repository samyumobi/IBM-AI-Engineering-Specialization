### Training Parameters in PyTorch
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> Your loss is a function of **w**. What method will calculate or accumulate gradients of your loss?
> 
> 1 / 1 point
> 
>  loss.grad 
> 

      loss.backward() 
> 
>  w.grad 
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
> What does the following line of code do :
> 
> w.data = w.data - lr * w.grad.data
> 
> 1 / 1 point
> 

      update parameters 
> 
>  zero the gradients before running the backward pass 
> 
>  calculate the iteration 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/CepfU/training-parameters-in-pytorch/attempt?redirectToCover=true#Tunnel Vision Close
