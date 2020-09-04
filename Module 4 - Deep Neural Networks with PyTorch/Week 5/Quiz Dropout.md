### Dropout
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> In what situation would you use dropout for classification
> 
> 1 / 1 point
> 

      your training accuracy is much larger then your test accuracy 
> 
>  your training accuracy is the same as test accuracy 
> 
> Check
> 
> Correct
> 
> correct, this is overfitting
> 
> 2.
> 
> Question 2
> 
> Consider the following Module or class :
> 
> class Net(nn.Module):
> 
>   def __init__(self, in_size, n_hidden, out_size, p)
> 
>     super(Net, self).__init__()
> 
>     self.drop=nn.Dropout(p=p)
> 
>     self.linear1=nn.Linear(in_size, n_hidden)
> 
>     self.linear2=nn.Linear(n_hidden, out_size)
> 
>   def forward(self, x):
> 
>     x=torch.relu(self.linear1(x))
> 
>     x=self.drop(x)
> 
>     x=self.linear2(x)
> 
>     return x
> 
> 
> how would you create a neural network with a dropout parameter of 0.9
> 
> 1 / 1 point
> 

      model =Net( in_size=10, n_hidden=100, out_size=10, p=0.9) 
> 
>  model =Net( in_size=0.9, n_hidden=100, out_size=10, p=10) 
> 
>  model =Net( in_size=0.9, n_hidden=0.9, out_size=10, p=10) 
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
> Select the constructer value to let 40% of the activations to the shut off
> 
> 1 / 1 point
> 

      nn.Dropout(0.4) 
> 
>  nn.Dropout(0.7) 
> 
> Check
> 
> Correct
> 
> incorrect
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/u3BHK/dropout/attempt?redirectToCover=true#Tunnel Vision Close
