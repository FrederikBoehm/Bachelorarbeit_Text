# Bachelor thesis "Is machine learning viable to predict shortterm market reactions after business report releases?"

Contains the `LaTeX` Code of the bachelor thesis.
See the whole thesis at [thesis.pdf](https://drive.google.com/file/d/13UToIcNns3IIFQclT3mDgzKHe3gXC1fK/view?usp=sharing).

## Kurzdarstellung
Den Text in Geschäftsberichten zu lesen und zu verstehen ist mit einem hohen Aufwand verbunden.
Diese thesis befasst sich mit einer Möglichkeit den Text automatisch zu verarbeiten, indem
ein Klassifikator vorhersagt, ob der Geschäftsbericht zu einer positiven oder negativen Marktreaktion
führt. Die Geschäftsberichte werden mit Hilfe von Bidirectional Encoder Representations from
Transformers (BERT) als Zahlenvektoren repräsentiert. Mit diesen und der Marktreaktion wird ein
Naive Bayes, ein kNN
Klassifikator, eine SVM und ein BLSTM trainiert. Die Klassifikationsgenauigkeit
von 69.3% mit der SVM spricht dafür, dass Vorhersagen am Aktienmarkt mit maschinellen
Lernverfahren möglich sind.

## Abstract
Reading and understanding the text of annual reports is related with a considerable amount of work.
This thesis examines a way to process the text automatically by predicting whether the report will lead
to a positive or negative market reaction. The business reports are embedded using BERT. With the
resulting embedding vectors and the market reactions a Naive Bayes, a kNN
classifier, a SVM and
a BLSTM were trained. The classification accuracy of 69.3% with the SVM leads to the conclusion
that stock market prediction with machine learning is possible.
