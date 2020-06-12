# Apache Spark and parallel data processing
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> rdd = sc.parallelize(range(100))
> 
> rdd2 = range(100)
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> Please consider the following code.
> 
> Where is the execution of API calls on "rdd" taking place?
> 
> 1 / 1 point 
> 

      In the ApacheSpark worker nodes 
> 
>  On the local Driver machine 
> 
> Check
> 
> Correct
> 
> Correct
> 
>  2.Question 2
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
>
> rdd = sc.parallelize(range(100))
> 
> rdd2 = range(100)
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> Please consider the following code.
> 
> Where is data in " **rdd2** " stored physically?
> 
> 1 / 1 point 
> 
>  In main-memory of ApacheSpark worker nodes 
> 

      On the local Driver machine 
> 
> Check
> 
> Correct
> 
> Correct
> 
>  3.Question 3
> 
> What is the parallel version of the following code?
> 
> <pre contenteditable="false" data-language="python" style="opacity: 1;" tabindex="0">
> 
> len(range(9999999999))
> 
> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
> 
> </pre>
> 
> 1 / 1 point 
> 

      sc.parallelize(range(9999999999)).count() 
> 
>  parallelize(range(9999999999)).count() 
> 
>  len(sc.parallelize(range(9999999999))) 
> 
>  size(sc.parallelize(range(9999999999))) 
> 
>  count(sc.parallelize(range(9999999999))) 
> 
> Check
> 
> Correct
> 
> Correct
> 
>  4.Question 4
> 
> Which storage solutions support seamless modification of schemas? (Select all that apply)
> 
> 1 / 1 point 
> 

      ObjectStorage 
> 
> Check
> 
> Correct
> 
> Correct
> 

      NoSQL 
> 
> Check
> 
> Correct
> 
> Correct
> 
>  SQL/Relational Databases 
> 
>  5.Question 5
> 
> Which storage solutions support dynamic scaling on storage? (Select all that apply)
> 
> 1 / 1 point 
> 

      ObjectStorage 
> 
> Check
> 
> Correct
> 
> Correct
> 

      NoSQL 
> 
> Check
> 
> Correct
> 
> Correct
> 
>  SQL/Relational Databases 
> 
>  6.Question 6
> 
> Which storage solutions support normalization and integrity checks on data out of the box? (Select all that apply)
> 
> 1 / 1 point 
> 
>  ObjectStorage 
> 
>  NoSQL 
> 

      SQL/Relational Databases 
> 
> Check
> 
> Correct
> 
> Correct
> 
>  7.Question 7
> 
> What is the advantage of using ApacheSparkSQL over RDDs? (select all that apply)
> 
> 1 / 1 point 
> 
>  ApacheSparkSQL bypasses the RDD interface which has been proven to be very complicated 
> 
>  SQL is simpler than RDD but has some performance drawbacks 
> 

      Catalyst and Tungsten are able to optimise the execution, so are more likely to execute more quickly than if you would had implemented something equivalent using the RDD API. 
> 
> Check
> 
> Correct
> 
> Correct
> 

      The API is simpler and doesn't require specific functional programming skills 
> 
> Check
> 
> Correct
> 
> Correct
>
> -- https://www.coursera.org/learn/machine-learning-big-data-apache-spark/exam/wJIiG/apache-spark-and-parallel-data-processing/attempt?redirectToCover=true#Tunnel Vision Close
