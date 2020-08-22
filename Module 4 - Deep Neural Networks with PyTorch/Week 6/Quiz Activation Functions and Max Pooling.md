## Activation Functions and Max Pooling
> 
> Total points 2
> 
> 1.
> 
> Question 1
> 
> Consider the following code:
> 
> z = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]])
> 
> What is the output of torch.relu(z)?
> 
> 1 point
> 
>  tensor([[1,0,-1],[2,0,-2],[1,0,-1]]) 
> 

      tensor([[1,0,0],[2,0,0],[1,0,0]]) 
> 
>  tensor([[0,0,-1],[0,0,-2],[0,0,-1]]) 
> 
> 2.
> 
> Question 2
> 
> Consider the following code:
> 
> 
> z = torch.tensor([[[1,2,3,-4],[0.0,2.0,-3.0,0],[0,2,3,1],[0,0,0,0]]])
> 
> max_ = torch.nn.MaxPool2d(2, stride=2)
> 
> What is the output of max_(z)?
> 
> 1 point
> 
>  tensor([[[[2,3,3],[2,3,3],[2,3,3]]]]) 
> 
>  tensor([[[[0,-3,-4],[0,-3,-3],[0,0,0]]]]) 
> 

      tensor([[[2., 3.],[2., 3.]]])
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/HCRYW/activation-functions-and-max-pooling/attempt#Tunnel Vision Close
