### 6.2 Softmax Prediction
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> Consider the following lines of code, what is yhat?
> 
> 
> z = torch.tensor([[10,5,0],[10,8,2],[10,5,1]])
> 
> _, yhat = z.max(1)
> 
> 1 / 1 point
> 
>  tensor([5,10,5]) 
> 

      tensor([0, 0, 0]) 
> 
>  tensor([1,0,0]) 
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
> We have two input features and four classes , what are the parameters for Softmax() constructor according to the above code?
> 
> 
> class Softmax (nn.Module):
> 
>     def __init__(self, in_size, out_size):
> 
>         super(Softmax, self).__init__()
> 
>         self.linear=nn.Linear(in_size, out_size)
> 
>     def forward(self, x):
> 
>         out=self.linear(x)
> 
>         return out
> 
> 
> 1 / 1 point
> 

      Sofmax(2,4) 
> 
>  Sofmax(4,4) 
> 
>  Sofmax(4,2) 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/YHxL2/6-2-softmax-prediction/attempt?redirectToCover=true#Tunnel Vision Close
