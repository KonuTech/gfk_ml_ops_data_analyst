# Measurement of served model bias

---


## Challenges and assumptions

---

> **Explainability**
> 
> should be taken into account during each stage of the ML lifecycle,
> for example, Problem Formation, Dataset Construction, Algorithm Selection,
> Model Training Process, Testing Process, Deployment, and Monitoring/Feedback.
> It is important to have the right tools to do this analysis.
> 
> Model explanation tools can help ML modelers and developers and other internal
stakeholders understand model characteristics as a whole prior to deployment
> and to debug predictions provided by the model once deployed
> 
> What is the function of an explanation in the machine learning context?
> An explanation can be thought of as the
answer to a why-question, thereby helping a human understand the cause of a prediction.
> In the context of a machine
learning model, we may be interested in answering questions such as
> “Why did the model predict a negative outcome
(e.g., loan rejection) for a given user?”, “How does the model make predictions?”,
> “Why did the model make an
incorrect prediction?”, and “Which features have the largest influence on the behavior of the model?”
> Thus, explanations can be useful for auditing and meeting regulatory requirements,
> building trust in the model and supporting human decision making, and debugging and improving model performance.
> 
> Some customers may care about contrastive explanations, or explanations of why an event X happened
> instead of some other event Y that did not occur. Here, X is the event that happened
> (an unexpected or surprising outcome as discussed above), and Y corresponds to an expectation
> based on their existing mental model. Note that for the same event X,
> different people may seek different explanations depending on their point of view or mental model Y.
> In the context of explainable AI, we can think of X as the example being explained and Y as a “baseline”
> that is typically chosen to represent an uninformative or average example in the dataset.
#### [SOURCE](https://pages.awscloud.com/rs/112-TZM-766/images/Amazon.AI.Fairness.and.Explainability.Whitepaper.pdf)

---
> **Bias Metrics**
>
> As various bias metrics examine different nuances and ways in which bias may arise,
> and there is not a single bias metric applicable across all scenarios,
> it is not always easy to know which ones apply in a particular situation or domain.
> Bias in a model arises in many ways. We provide a sixcategory taxonomy of sources of bias:
> 
> 
> 1. Biased labels. This arises from human biases and accumulates in datasets. 
> It is particularly prevalent in public datasets with multiple labelers, like police data,
> public opinion datasets, etc., see Wauthier and Jordan (2011)
> 
> 
> 2. Biased features, also known as “curation” bias. Here, bias arises from selecting some features and dropping 
> others and can occur directly or indirectly. For example, in lending,
> a modeler may choose features that are more likely to disadvantage one group and leave out features
> that would favor that group. While this may be deliberate,
> it is also possible to have these be done as part of an unconscious process.
> O’Neil (2016) gives a great example where her model for why children love eating their vegetables
> was an outcome of culinary curation, where they seem to eat all their vegetables
> given no servings of pizza, potatoes, meat, etc.
> 
> 
> 3. Objective function bias, noted by Menestrel and Wassenhove (2016).
> One case in which this occurs is when the loss function may be overly focused on outliers
> and if outliers are of specific types in the dataset, the modeler may inject bias.


#### [SOURCE](https://pages.awscloud.com/rs/112-TZM-766/images/Fairness.Measures.for.Machine.Learning.in.Finance.pdf)
