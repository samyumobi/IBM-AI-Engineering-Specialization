### Neural Networks
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> Consider the following neural network model or class:
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
> How many hidden neurons does the following neural network object have?
>
> 
> model=Net(1,6,1)
> 
> 
> 1 / 1 point
> 

     6
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
> What does the following line of code do?<<
> 
> 
> torch.sigmoid(self.linear1(x))
> 
> 
> 1 / 1 point
> 
>  creates a linear object 
> 

      Applies a sigmoid activation to every element of the tensor x 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/0jvUs/neural-networks/attempt?redirectToCover=true#Tunnel Vision Close
