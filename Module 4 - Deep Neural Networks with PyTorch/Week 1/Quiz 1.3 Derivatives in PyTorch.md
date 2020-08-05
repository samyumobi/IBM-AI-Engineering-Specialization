### 1.3 Derivatives in PyTorch
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> What task does the following lines of code perform?
> 
> 
> q=torch.tensor(1.0,requires_grad=True)
> 
> fq=2q**3+q
> 
> fq.backward()
> 
> q.grad
>  
> 1 / 1 point 
> 

      Determines the derivative of **2q**3+q** at **q=1** 
> 
>  Makes a function that we can use in any part of the code 
> 
>  Differentiates the function with respect to all values 
> 
> Check
> 
> Correct
> 
>  2.Question 2
> 
> What's wrong with the following lines of code?
>  
> q=torch.tensor(1.0,requires_grad=False)
> 
> fq=2q**3+q
> 
> fq.backward()
> 
> q.grad
>  
> 1 / 1 point 
> 

      The parameter requires_grad should be set to True 
> 
>  q is a float 
> 
>  A differentiable function should be used 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/r64SB/1-3-derivatives-in-pytorch/attempt?redirectToCover=true#Tunnel Vision Close
