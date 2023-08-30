import cv2
import numpy as np



list_img = cv2.imread('list_img3.jpg', 1)
list_img  = cv2.cvtColor(list_img, cv2.COLOR_BGR2GRAY)
black_spot = cv2.imread('black_spot.jpg', 0)
# black_spot = cv2.cvtColor(black_spot, cv2.COLOR_BGR2GRAY)
mark1 = cv2.imread('mark1-2.jpg', 0)
# mark1 = cv2.cvtColor(mark1, cv2.COLOR_BGR2GRAY)
mark2 = cv2.imread('mark2-2.jpg', 0)
# mark2 = cv2.cvtColor(mark2, cv2.COLOR_BGR2GRAY)

# cv2.imshow('list', list_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

cv2.imshow('spot', black_spot)
cv2.waitKey()
cv2.destroyAllWindows()

result = cv2.matchTemplate(list_img, black_spot, cv2.TM_CCOEFF_NORMED)
mark1_res = cv2.matchTemplate(list_img, mark1, cv2.TM_CCOEFF_NORMED)
mark2_res = cv2.matchTemplate(list_img, mark2, cv2.TM_CCOEFF_NORMED)
# print(result)
# cv2.imshow('result', result)
# cv2.waitKey()
# cv2.destroyAllWindows()

retval, threshold = cv2.threshold(list_img, 120, 255, cv2.THRESH_BINARY)
retval_spot, spot_threshold = cv2.threshold(black_spot, 80, 255, cv2.THRESH_BINARY)

result = cv2.matchTemplate(threshold, spot_threshold, cv2.TM_CCOEFF_NORMED)

cv2.imshow('list', spot_threshold)
cv2.waitKey()
cv2.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mark1_res)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mark2_res)
# print(max_loc)
# print(max_val)

w = black_spot.shape[1]
h = black_spot.shape[0]

w_m1 = mark1.shape[1]
h_m1 = mark1.shape[0]

w_m2 = mark2.shape[1]
h_m2 = mark2.shape[0]

# cv2.rectangle(list_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)

# cv2.imshow('list', list_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

treshhold = .90

yloc, xloc = np.where(result >= .90)
print(len(xloc))
# print(xloc)
# print(yloc)

yloc_m1, xloc_m1 = np.where(mark1_res >= treshhold)
yloc_m2, xloc_m2 = np.where(mark2_res >= treshhold)

answers = {}

for i, (x, y) in enumerate(zip(xloc, yloc), 1):
    # print(f'spot{(x, y)}')
    answers[i] = (x, y)
    cv2.rectangle(threshold, (x, y), (x + w, y + h), (0, 255, 255), 2)
print(f'answers: {answers}')
options = {}

for i, (x, y) in enumerate(zip(xloc_m1, yloc_m1), 1):
    # print(f'm1{(x, y)}')
    options[i] = x
    cv2.rectangle(list_img, (x, y), (x + w_m1, y + h_m1), (0, 255, 255), 2)

print(f'options: {options}')

questions = {}
for i, (x, y) in enumerate(zip(xloc_m2, yloc_m2), 1):
    # print(f'm2{(x, y)}')
    questions[i] = y
    cv2.rectangle(list_img, (x, y), (x + w_m2, y + h_m2), (0, 255, 255), 2)
print(f'questions: {questions}')

# cv2.rectangle(list_img, (123, 34), (123 + w_m1, 34 + h_m1), (0, 255, 255), 2)

cv2.imshow('list', threshold)
cv2.waitKey()
cv2.destroyAllWindows()

options = {'a': 50, 'b': 80, 'c': 110, 'd': 150}

