## Simple Dataset
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> What is wrong with the following class:
> 
> 
> class toy_set(Dataset):
> 
>     # Constructor with defult values 
> 
>     def __init__(self, length = 100, 
> 
>       transform = None):
> 
>         self.len = length
> 
>         self.x = 2 * torch.ones(length, 
> 
>           2)
> 
>         self.y = torch.ones(length, 1)
> 
>         self.transform = transform
> 
>     # Getter
> 
>     def __getitem__(self, index):
> 
>         sample = self.x[index], self
> 
>           .y[index]
> 
>         if self.transform:
> 
>             sample = self.transform
> 
>               (sample)     
> 
>         return sample
> 
>     # Get Length
> 
>     def __len__(self):
> 
> 
> 1 / 1 point 
> 

      the method __len__(self) does not return anything 
> 
>  it did not subclass Dataset 
> 
>  there is no __getitem__ method 
> 
> Check
> 
> Correct
> 
> correct
> 
>  2.Question 2
> 
> Which of the following are the build-in functions you need to define while customizing the class for transforming?
> 
> 1 / 1 point 
> 

      __init__ 
> 
> Check
> 
> Correct
> 
> correct
> 
>  np.array() 
> 

      __call__ 
> 
> Check
> 
> Correct
> 
> correct
>
> -- https://www.coursera.org/learn/deep-neural-networks-with-pytorch/exam/ZvoTr/simple-dataset/attempt?redirectToCover=true#Tunnel Vision Close
