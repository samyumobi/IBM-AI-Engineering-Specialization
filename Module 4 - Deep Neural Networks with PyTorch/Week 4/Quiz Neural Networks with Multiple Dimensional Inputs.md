## Neural Networks with Multiple Dimensional Inputs
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> True or False? The following dataset is linearly separable?
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/tLqJ5urYEem0mBK9yWeCVg_b6245ac729f0efbc84ce51b723790585_linearly_sep_data-1-.png?expiry=1598918400000&hmac=b3NROlQ7dZTtNUKMMKknCBrhxfPznljChWm0sGW9RBk)
> 
> 1 / 1 point
> 

    False 
> 
>  Ture 
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
> How many dimensions is the input for the following neural network object:
> 
> 
> class Net(nn.Module):
> 
>     def __init__(self,D_in,H,D_out):
> 
>         super(Net,self).__init__()
> 
>         self.linear1=nn.Linear(D_in,H)
> 
>         self.linear2=nn.Linear(H,D_out)
> 
>     def forward(self,x):
> 
>         x=torch.sigmoid(self.linear1(x))  
> 
>         x=torch.sigmoid(self.linear2(x))
> 
>         return x
> 
> model=Net(4,10,1)
> 
> 
> 1 / 1 point
> 

     4
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/Q7mhT/neural-networks-with-multiple-dimensional-inputs/attempt?redirectToCover=true#Tunnel Vision Close
