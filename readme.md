# Text Classification (Daisi Hackathon)

Python function as a web service to classify and extract features from text according to 4 different categories.

Text classification according to the following main categories:
* Business
* Technology
* Politics
* Entertainment

It will extract features from text and classify it using pretrained machine learning model on **BBC News** articles dataset. 

**Example**

`text = "The next Prime Minister will not be elected by the British public"`

Classifying this text will return **Politics**.

`text = "Currently, when switching a Wear OS smartwatch to a new Android smartphone, users are required to factory reset the watch."`

Also, this text will return **Technology**.

### How to call it

* Load the Daisi
<pre>
import pydaisi as pyd
text_classification_fx = pyd.Daisi("oghli/Text Classification FX")
</pre>

* Call the `text_classification_fx` end point, passing the input text to classify it
<pre>
text = "Buying and selling stocks and shares has always involved a lot of third parties, such as brokers and the stock exchange itself. Here is how trading works"
text_classify = text_classification_fx.classify_text(text).value
text_classify
</pre>
* it will return classification result

  `{0: 'business'}`

Check another call:
<pre>
text = "From light planes to wide-body jets, fly highly detailed and accurate aircraft in the next generation of Microsoft Flight Simulator. Test your piloting skills against the challenges of night flying, real-time atmospheric simulation and live weather in a dynamic and living world."
text_classify = text_classification_fx.classify_text(text).value
text_classify
</pre>
it will return

`{3: 'entertainment'}`

Function `st_ui` included in the app to render the user interface of the application endpoints.

For more info check Daisi documentation: 
https://doc.daisi.io/

