### Deep Neural Networks
> 
> Total points 2
> 
> 1.
> 
> Question 1
> 
> What kind of activation function is being used in the second hidden layer:
> 
> 
> class NetTanh(nn.Module):
> 
>     # Constructor
> 
>     def __init__(self, D_in, H1, H2, D_out):
> 
>         super(NetTanh, self).__init__()
> 
>         self.linear1 = nn.Linear(D_in, H1)
> 
>         self.linear2 = nn.Linear(H1, H2)
> 
>         self.linear3 = nn.Linear(H2, D_out)
> 
>     # Prediction
> 
>     def forward(self, x):
> 
>         x = torch.sigmoid(self.linear1(x))
> 
>         x = torch.tanh(self.linear2(x))
> 
>         x = self.linear3(x)
> 
>         return x
> 
> 1 point
> 

      tanh 
> 
>  sigmoid 
> 
> 2.
> 
> Question 2
> 
> Consider the following code:
> 
> 
> class Net(nn.Module):
> 
>     # Constructor
> 
>     def __init__(self, D_in, H1, H2, D_out):
> 
>         super(Net, self).__init__()
> 
>         self.linear1 = nn.Linear(D_in, H1)
> 
>         self.linear2 = nn.Linear(H1, H2)
> 
>         self.linear3 = nn.Linear(H2, D_out)
> 
>     # Prediction
> 
>     def forward(self,x):
> 
>         x = torch.sigmoid(self.linear1(x)) 
> 
>         x = torch.sigmoid(self.linear2(x))
> 
>         x = self.linear3(x)
> 
>         return x
> 
> model = Net(3,5,4,1)
> 
> 
> How many hidden layers are there in this model?
> 
> 1 point
> 
> Enter answer here
>

    2
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/riOL7/deep-neural-networks/attempt#Tunnel Vision Close
