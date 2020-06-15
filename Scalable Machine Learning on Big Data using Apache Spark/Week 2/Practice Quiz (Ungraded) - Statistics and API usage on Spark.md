# Practice Quiz (Ungraded) - Statistics and API usage on Spark
> 
> Total points 2
> 
>  1.Question 1
> 
> Why is it useful to compute statistical moments?
> 
> 1 / 1 point 
> 

      Statistical moments tell us a lot about the value distribution of different features (table columns). 
> 
>  Derived values of statistical moments are used as input for downstream machine leaerning algorithms 
> 
> Check
> 
> Correct
> 
> Correct
> 
>  2.Question 2
> 
> Which API should I use? RDD, DataFrame or SQL?
> 
> 1 / 1 point 
> 

      You should always use the DataFrame or SQL API because it makes use of the Tungsten and Catalyst optimizers resulting in faster running code and less development effort. RDDs exists under the hood and are only useful if a problem can't be expressed using the DataFrame or SQL API 
> 
>  RDD is always the preferred way to go because applying functional programming is the way to make any program run fastest. The DataFrame or SQL API are only for less skilled users until they make their way through to use RDDs 
> 
> Check
> 
> Correct
> 
> Correct
>
> -- https://www.coursera.org/learn/machine-learning-big-data-apache-spark/quiz/9uLsr/practice-quiz-ungraded-statistics-and-api-usage-on-spark/attempt?redirectToCover=true#Tunnel Vision Close
