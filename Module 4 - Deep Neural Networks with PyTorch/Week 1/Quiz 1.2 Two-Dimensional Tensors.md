### 1.2 Two-Dimensional Tensors
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> How do you convert the following Pandas Dataframe to a tensor:
> 
> 
> df = pd.DataFrame({'A':[11, 33, 22],'B':[3, 3, 2]})
> 
> 
> 1 / 1 point 
> 

      torch.tensor(df.values) 
> 
>  torch.tensor(df) 
> 
> Check
> 
> Correct
> 
> correct
> 
>  2.Question 2
> 
> What is the result of the following:
> 
> 
> X = torch.tensor([[1, 0], [0, 1]])
> 
> Y = torch.tensor([[2, 1], [1, 2]]) 
> 
> X_times_Y = X * Y
> 
> 
> 1 / 1 point 
> 

      tensor([[2, 0], [0, 2]]) 
> 
>  tensor([[0, 1], [1, 4]]) 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/kYl3o/1-2-two-dimensional-tensors/attempt?redirectToCover=true#Tunnel Vision Close
