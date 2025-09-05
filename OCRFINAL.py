import pytesseract
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Display function
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width  = im_data.shape[:2]    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image.
    ax.imshow(im_data, cmap='gray')
    plt.show()

image_file = "D:/DIPLOMA/3rdSEMMP/DSP/d1.jpg"
img = cv2.imread(image_file)

# Inverted image
inverted_image = cv2.bitwise_not(img)
# cv2.imwrite("inverted.jpg", inverted_image)
# display("inverted.jpg")

# Binarization
def binarize(image, threshold=200):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 13)
    return im_bw
binary_image = binarize(img)
cv2.imwrite("binary_image.jpg", binary_image)
# display("binary_image.jpg")

#Deskewing
edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
angle_sum = 0
valid_lines = 0
for rho, theta in lines[:, 0]:
    if 30 <= np.degrees(theta) <= 150:
        valid_lines += 1
        angle_sum += np.degrees(theta)
average_angle = angle_sum / valid_lines
rotation_matrix = cv2.getRotationMatrix2D((lines.shape[1] // 2, lines.shape[0] // 2), -average_angle, 1)
deskewed_image = cv2.warpAffine(lines, rotation_matrix, (lines.shape[1], lines.shape[0]), flags=cv2.INTER_LINEAR)
# cv2.imwrite("desk.jpg", deskewed_image)

import cv2
#import PIL.Image
import pytesseract
from pytesseract import Output
#import string
from matplotlib import pyplot as plt

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width  = im_data.shape[:2]    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image.
    ax.imshow(im_data, cmap='gray')
    plt.show()

"""
Page segmentation modes:
0   Orientation and Script Detection (OSD) only.
1   Automatic page segmentation with OSD.
2   Automatic page segmentation, but no OSD, or OCR.
3   Fully automatic page segmentation, but no OSD (Default).
4   Assume a single column of text of variable sizes.
5   Assume a single uniform block of vertically aligned text .
6   Assume a single uniform block of text.
7   Treat the image as single text line.
8   Treat the image as single word.
9   Treat the image as single word in a circle.
10  Treat the image as a single character.
11  Sparse text. Find as much text as possible in no particular order.
12  Sparse text with OSD.
13  Raw line. Treat the image as single text line, bypassing hacks that are Tesseract-specific.
"""
"""
OCR engine modes:
0   Legacy engine only.
1   Neural nets LSTM (Long Short Term memory) engine only.
2   Legacy + LSTM engines.
3   Default, based on what is available.
"""

myconfig = r"--psm 12 --oem 3"
myconfig2 = r"--psm 6 --oem 3"
img = cv2.imread("binary_image.jpg")
height , width , _ = img.shape

data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)
print(data['text'])

extracted_text = pytesseract.image_to_string(img, config=myconfig2)
f=open("stroage.txt","w+")
a=f.write(extracted_text)
f.close

amount_boxes = len(data['text'])
for i in range(amount_boxes):
    if float(data['conf'][i])>80:
        (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x, y), (x+width, y+width), (0, 255, 0), 2)
        img = cv2.putText(img, data['text'][i], (x, y+height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
print("")
print(extracted_text)
cv2.imwrite("Image.jpg", img)
display("Image.jpg")


#using mysql connector
import mysql.connector

# Function to read the content of the notepad file
def readcontent(stroage):
    with open(stroage, 'w+') as file:
        content = file.read()
        print(content)
    return content

# Function to create a table in the database
def create_table(cursor):
    cursor.execute('CREATE TABLE IF NOT EXISTS notepad1 (id INT AUTO_INCREMENT PRIMARY KEY, content TEXT)')

# Function to insert notepad content into the database
def insert_notepad_content(cursor, content1):
    cursor.execute('INSERT INTO notepad1 (content) VALUES (%s)', (content1,))
    print(content1,'hii')
    mydb.commit()
    
if __name__ == '__main__':
    notepad_content = readcontent("D:/DIPLOMA/3rdSEMMP/DSP/stroage.txt")  # Replace with the path to your notepad file
   # notepad_content = readcontent(storage)
    print(notepad_content)
    
    # MySQL connection configuration
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="mydatabase" )
#mycursor= mydb.cursor()
    # Connect to the MySQL database
    #conn = mysql.connector.connect(**mydb)
    cursor = mydb.cursor()
 
    create_table(cursor)
    insert_notepad_content(cursor, extracted_text)
    # Close the database connection
    cursor.close()
    mydb.close()



# showing sql data

# Function to retrieve and display the paragraph from the database
def displaying(cursor):
    cursor.execute("SELECT content FROM notepad")
    result = cursor.fetchone()

    if result:
        paragraph = result[0]
        print("Paragraph from the database:")
        print("Here is the paragraph store in DATABASE")
        print(paragraph)
    else:
        print("No paragraph found in the database.")

if __name__ == '__main':
    # MySQL connection configuration
    mydb1 =mysql.connector.connect(
        host ='localhost',
        user='root',
        password= '',
        database= 'mydatabase'
    )

    # Connect to the MySQL database
    
    cursor = mydb1.cursor()

    displaying(cursor)

    # Close the database connection
    cursor.close()
    mydb1.close()
