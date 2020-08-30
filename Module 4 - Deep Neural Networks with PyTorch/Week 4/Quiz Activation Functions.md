### Activation Functions
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> Usually, what activation function would you use if you had more than 10 hidden layers?
> 
> 1 / 1 point
> 
>  Sigmoid 
> 

      Relu 
> 
>  Tanh 
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
> What is the problem with the tanh and sigmoid activation function?
> 
> 1 / 1 point
> 
>  They are discontinuous functions 
> 
>  You can't take the derivative 
> 

    The derivative is near zero in many regions 
> 
>  They are periodic functions 
> 
> Check
> 
> Correct
> 
> correct
> 
> 3.
> 
> Question 3
> 
> What activation function is used in the following class
> 
> 
> class NetRelu(nn.Module):    
> 
>   def __init__(self,D_in,H,D_out):                                 super(NetRelu
> 
>     ,self).__init__() 
> 
>                    self.linear1=nn.Linear(D_in,H)        
> 
>            self.linear2=nn.Linear(H,D_out)            
> 
>           def forward(self,x):        
> 
>           x=torch.relu(self.linear1(x)))          
> 
>           x=self.linear2(x)     
> 
>             return x
> 
> 1 / 1 point
> 

      relu 
> 
>  tanh 
> 
>  Sigmoid 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/Ue03I/activation-functions/attempt?redirectToCover=true#Tunnel Vision Close
