# Course-Match
2020 Summer Research Project Repo

Hartline et. al designed the [Random Partial Improvement Mechansim](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975994.129) to approximate the [Nash Bargaining Solution](https://en.wikipedia.org/wiki/Cooperative_bargaining) (NBS). My summer project was to implement this mechanism and to modify it to be used specifically for assigning students course schedules based on their preferences.

## Background
RPI  is  designed  to  approximate  the  Nash  bargaining  solution,  which  takes  preferencesfrom agents and provides an outcome that optimizes fairness and efficiency for each agent. A  student’s  preferences  would  be  values  for  how  much  she  wants  to  take  each  courseavailable to her.  While fairness and efficiency are desired qualities of a mechanism, theNash  bargaining  solution  is  not  truthful;  a  student  reporting  her  course  preferences  toa truthful course match mechanism would have the best chance of getting her preferredcourses if she reports her real preferences. A course match mechanism should be truthfulso that it is easy and intuitive for the students to try to get the classes they want. The performance of RPI is evaluated by the worst-case ratio of a student’s utility out-putted from RPI to the same student’s utility as a result of the Nash bargaining solution.Student utility in this case is the expected value the student has for her schedule.  RPI’struthfulness and close approximation to the Nash bargaining solution make it an excellentbase to design a course match-specific mechanism from.

## Outcome
The implementation of the mechanism is complete; however, the use of a convex optimization engine is required which seems to scale of the program. This has stopped me from finding the optimal value of a parameter k which is the ratio of uniformly distributed course capacity to [Partial Allocation Mechanism](https://epubs.siam.org/doi/pdf/10.1137/15M1053682?casa_token=2TKCUYAltCMAAAAA:IH2Ua_zGds8PKyiK1J8xMDU74WefaUlNowL9UoQtgXA_gnQW5Lhhtaj3XuzvkGP59cldXTsdVjRXTA). 
I intended to find the optimal k using a monte carlo simulation framework; however, when using Python's threading capabilities with Northwestern University's Quest high performance computing cluster with sample inputs of 52 students and 32 course options to the tweaked RPI function, computing just 100 iterations of this takes nearly a week. A proper approximation is so far intractable. I am currently trying different ways to find this k value.

## RPI Program
The tweaked RPI program can be found in RPI_Tweaked_Cap.py under the RPI function. All of the helper functions and their descriptions are also included in this file. 
