> #### TORCH-VISION MODELS
> 
> Latest Submission Grade
> 
> 100%
> 
> 1.
> 
> Question 1
> 
> What do the following lines of code do:
> 
> 
> model = models.densenet121(pretrained=True)
> 
> for param in model.parameters():
> 
>     param.requires_grad=False
> 
> 1 / 1 point
> 

      The following lines of code will set the attribute requires_grad to False. As a result, the parameters will not be affected by training. 
> 
>  set the number of classes for a pertained model 
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
> We would like to train our pre-trained model "mode"l whats wring with the following lines of code:
> 
> for epoch in range(n_epochs):
> 
>     for x, y in train_loader:
> 
>         model.train() 
> 
>         #clear gradient 
> 
>         optimizer.zero_grad()
> 
>         #make a prediction 
> 
>         z=model(x)
> 
>         # calculate loss 
> 
>         loss=criterion(z,y)
> 
>         # calculate gradients of parameters 
> 
>         loss.backward()
> 
>         # update parameters 
> 
>         optimizer.step() 
> 
> 
> 1 / 1 point
> 

      we did not set the model to train 
> 
>  nothing 
> 
>  you did not clear the gradient 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/M7cTS/torch-vision-models/attempt?redirectToCover=true#Tunnel Vision Close
