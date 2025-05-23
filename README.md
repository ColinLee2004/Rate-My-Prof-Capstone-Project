Description of dataset: The datafile rmpCapstoneNum.csv contains 89893 records. Each of these
records (rows) corresponds to information about one professor.
The columns represent the following information, in order:

1: Average Rating (the arithmetic mean of all individual quality ratings of this professor)

2: Average Difficulty (the arithmetic mean of all individual difficulty ratings of this professor)

3: Number of ratings (simply the total number of ratings these averages are based on)

4: Received a “pepper”? (Boolean - was this professor judged as “hot” by the students?)

5: The proportion of students that said they would take the class again

6: The number of ratings coming from online classes

7: Male gender (Boolean – 1: determined with high confidence that professor is male)

8: Female (Boolean – 1: determined with high confidence that professor is female

With this dataset in hand, we would like you to answer the following questions:

1. Activists have asserted that there is a strong gender bias in student evaluations of professors, with
male professors enjoying a boost in rating from this bias. While this has been celebrated by ideologues,
skeptics have pointed out that this research is of technically poor quality, either due to a low sample
size – as small as n = 1 (Mitchell & Martin, 2018), failure to control for confounders such as teaching
experience (Centra & Gaubatz, 2000) or obvious p-hacking (MacNell et al., 2015). We would like you to
answer the question whether there is evidence of a pro-male gender bias in this dataset.
Hint: A significance test is probably required.

2. Is there an effect of experience on the quality of teaching? You can operationalize quality with
average rating and use the number of ratings as an imperfect – but available – proxy for experience.
Again, a significance test is probably a good idea.

3. What is the relationship between average rating and average difficulty?
   
4. Do professors who teach a lot of classes in the online modality receive higher or lower ratings than
those who don’t? Hint: A significance test might be a good idea, but you need to think of a creative but
suitable way to split the data.

5. What is the relationship between the average rating and the proportion of people who would take
the class the professor teaches again?

6. Do professors who are “hot” receive higher ratings than those who are not? Again, a significance
test is indicated.

7. Build a regression model predicting average rating from difficulty (only). Make sure to include the R2
and RMSE of this model.

8. Build a regression model predicting average rating from all available factors. Make sure to include
the R2 and RMSE of this model. Comment on how this model compares to the “difficulty only” model
and on individual betas. Hint: Make sure to address collinearity concerns.

9. Build a classification model that predicts whether a professor receives a “pepper” from average
rating only. Make sure to include quality metrics such as AU(RO)C and also address class imbalances.

10. Build a classification model that predicts whether a professor receives a “pepper” from all available
factors. Comment on how this model compares to the “average rating only” model. Make sure to
include quality metrics such as AU(RO)C and also address class imbalances.

Extra credit: Tell us something interesting about this dataset that is not trivial and not already part of
an answer (implied or explicitly) to these enumerated questions [Suggestion: Do something with the
qualitative data, e.g. major, university or state by linking the two data files
