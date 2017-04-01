About Hanafuda
==============

Hanafuda are traditional japanese playing cards that are used to play a number of games. There are twelve suits, representing months (with four cards of every suit). Each month is designated by a different flower. There are two common and two special cards (with additional design) in every suit. Usually one of the special cards is designed with red or blue ribbon. Every card in pack is unique.

Example of cards representing Januar:

![](https://github.com/aleker/PyHanafudaClassificator/blob/master/pictures/01-01pkt-0r-a.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/pictures/01-01pkt-0r-b.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/pictures/01-05pkt-1r.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/pictures/01-20pkt-0r.jpg)

Used technologies
=================
Program is written in _Python 3.5.0_. 

_OpenCV_ library was used for image processing.

Decision tree from _scikit-learn_ was used as card classifier.

Hanafuda Classifier
=======================

Processing training cards
-------------------------
As input program receives photo of single card. First of all, function 

```
(colorful_count, masks) = find_colour_count(image, file_name)
```
is called. This function retrieves colours (white, red and blue) from card pictures creating three masks. It also returns number of coloured pixels in every mask.  

Example of card image and sequentially blue, red and white mask.

![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/10-05pkt-2r.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/10-05pkt-2rb.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/10-05pkt-2rr.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/10-05pkt-2rw.jpg)

Next, function

```
(hu_moments_for_all_colours) = compute_hu_moments(image, file_name)
```
is called. This function computes seven _Hu moments_ for every colour mask. As a result it returns array of three arrays with seven image moments.

Farther, all of the computed information and fact of owning ribbon are given to decision tree as training data.

Processing testing images
-------------------------

As input program receives photo of cards. Application separates every card from picture using dilation and edge detection.  
Then program scales and turns them so they will be ready for image processing. 

![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/IMG_20161210_112547380.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/P1110883.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/P1110886.jpg)

Than function

```
( _ , rebbons_colour) = findRibbon(image, file_name)
```
is called.

![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/IMG_20161210_112743967.jpg =250x)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/0IMG_20161210_112743967b.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/0IMG_20161210_112743967.jpg)

Results
-------

This project focused on image processing, not machine learning. For this reason decision tree has't got more training cards than one of every type (in sum, 48 cards) so results was halfway correct.
This is the example. On the left side are card images cut out from original image.
On the right side of every founded card are cards assigned by classifier.

INPUT:

![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/P11108867.jpg)

OUTPUT:

![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/0P1110886.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/1P1110886.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/2P1110886.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/3P1110886.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/4P1110886.jpg)
![](https://github.com/aleker/PyHanafudaClassificator/blob/master/processing_examples/5P1110886.jpg)
