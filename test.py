
# Import required packages
import cv2
import pytesseract
import re
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
# Mention the installed location of Tesseract-OCR in your system
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


# Read image from which text needs to be extracted

img = cv2.imread("kientest1.png")
img = cv2.resize(img,(700,500))

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (thresh, img) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

cv2.imshow('abc', thresh1)
cv2.waitKey()
# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9 ))

# Appplying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
cv2.imshow('dmm',dilation)
cv2.waitKey()
# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

#Vierw image img


# Creating a copy of image
im2 = img.copy()

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
configuration = ("-l eng --oem 1 --psm 8")
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x-5, y-5), (x + w+ 5, y + h+5), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = im2[y-5:y + h+5, x-5:x + w+5]
    # Open the file in append mode
    # Apply OCR on the cropped image
    
    #Lines.append(format(text))
    # Appending the text into file
    # cv2.imshow('ct', cropped)
    # cv2.waitKey()
   
    cv2.imshow('1', rect)
    cv2.waitKey()
    

cv2.imshow('abcd',im2)
cv2.waitKey()