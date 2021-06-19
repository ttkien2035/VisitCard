
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
with open("myfile.txt", "w+", encoding="utf-8") as f:
        f.write("\n")
        f.close

# Read image from which text needs to be extracted
img = cv2.imread("testkien.png")
img = cv2.resize(img,(700,500))

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (thresh, img) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# cv2.imshow('abc', thresh1)
# cv2.waitKey()
# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# Appplying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

#Vierw image img


# Creating a copy of image
im2 = img.copy()
Lines=[]
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
    text = pytesseract.image_to_string(cropped, lang='eng')
    #Lines.append(format(text))
    # Appending the text into file
    # cv2.imshow('ct', cropped)
    # cv2.waitKey()
    print(text)
    cv2.imshow('1', rect)
    cv2.waitKey()
    with open("myfile.txt", "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
        f.close
def checkemail(email):
    # pass the regular expression
    # and the string in search() method
    regex = '([@]|(^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$) )'
    if(re.search(regex, email)):
        return True
    else:
        return False
    

def checkphone(e):
    # pass the regular expression
    # and the string in search() method
    regex = '(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}|\d{4}[-\.\s]??\d{3})'
    
    if(re.search(regex, e)):
        return True
    else:
        return False
def checkweb(e):
    # pass the regular expression
    # and the string in search() method
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))|(www+)|(://+)"     
    if(re.search(regex, e)):
        return True
    else:
        return False
def checkAddress(e):
    regex = r"(\d{1,4}( \w+){1,5}, (.*), ( \w+){1,5}, (AZ|CA|CO|NH), [0-9]{5}(-[0-9]{4})?)|(add+)|(Add+)|(Dia+)"
    if(re.search(regex,e)):
        return True
    else:
        return False
def checkName(e):
    regex = r"((^[A-z]{2,10}[\s]+[A-z]{3,10}[\s]+[A-z]{3,10}[\s]+[A-z]{3,10}$)|(^[A-z]{2,10}[\s]+[A-z]{3,10}[\s]+[A-z]{3,10}$)|(^[A-z]{3,10}[\s]+[A-z]{3,10}$))"
    if(re.search(regex,e)):
        return True
    else:
        return False
file = open('myfile.txt', 'r',encoding="utf-8")
Lines = file.readlines()

for line in Lines:
    if line.strip()=='' or line.strip()=='/n':
        Lines.remove(line)
# for line in Lines:
#     print(line)
# for line in Lines:
#     doc = nlp(line)
#     print([(X.text, X.label_) for X in doc.ents])
print("Name Found :")
for line in Lines:
    doc = nlp(line)
    for X in doc.ents:
        if X.label_ == 'PERSON':
            print("            ",X.text)
print("Email Found :")
for line in Lines:
    if checkemail(line.strip()):
        print("            ", line.strip())
print("Phone Found :")
for line in Lines:
    if checkphone(line.strip()):
        print("            ",line.strip())
print("Website Found :")
for line in Lines:
    if checkweb(line.strip()):
        print("            ",line.strip())
print("Address Found :")
for line in Lines:
    if checkAddress(line.strip()):
        print("            ",line.strip())
cv2.imshow('abcd',im2)
cv2.waitKey()