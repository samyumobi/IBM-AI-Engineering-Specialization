## 9.1 Convolution
> 
> Total points 2
> 
> 1.
> 
> Question 1
> 
> How would you create a convolution object with a kernel size of 3
> 
> 1 point
> 
> 

     conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
>
> 
> conv = nn.Conv2d(in_channels=3, out_channels=3,kernel_size=2)
> 
> 
> 2.
> 
> Question 2
> 
> Consider the following code:
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> 1
> 
> cov=nn.conv2d(in_channels=3,out_channels=1,kernel_size=2,stride=3,padding=1)
>
> 
> how many rows and columns will be padded?
> 
> 1 point
> 

     2
> 
> Enter math expression here
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/HdxEQ/9-1-convolution/attempt#Tunnel Vision Close
