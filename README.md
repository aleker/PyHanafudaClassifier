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

Hanafuda Classificactor
=======================

Processing training cards
-------------------------
As input program receives photo of single card. First of all, function 

```
(colorful_count, masks) = find_colour_count(image, file_name)
```
is called. This function retrieves colours (white, red and blue) from card pictures creating three masks. It also returns number of coloured pixels in every mask.  

Next, function

```
(hu_moments_for_all_colours) = compute_hu_moments(image, file_name)
```
is called. This function computes seven _Hu moments_ for every colour mask. As a result it returns array of three arrays with seven image moments.

Farther, all of the computed information and fact of owning ribbon are given to decision tree as training data.

Processing testing images
-------------------------

As input program receives photo of cards. Application separates every card from picture, scales and turns them so thay will be ready for image processing. Than function

```
( _ , rebbons_colour) = findRibbon(image, file_name)
```
is called.
