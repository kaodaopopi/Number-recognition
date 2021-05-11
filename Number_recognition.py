# 導入函式庫
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from matplotlib import pyplot as plt
import cv2

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_test.shape)
print(y_test.shape)

#cv2.imwrite('C:/Users/User/Desktop/output.jpg', X_test[1])

# 建立簡單的線性執行的模型
model = Sequential()

# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
# Add output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

# 將 training 的 input 資料轉為2維
# 圖片是 28 28 的，接下來也要把我們要預測的圖片的大小也要是 28*28
X_train_2D = X_train.reshape(60000, 28*28).astype('float32')
X_test_2D = X_test.reshape(10000, 28*28).astype('float32')

print("X_test_2D: ",X_test_2D.shape)

#做正規化，補充一下，這個選擇除以255，是因為255 就是陣列中會出現的最大值，所以它希望值變得相對簡單一點，
#變成 0~1之間，原本是 0~255 ，所以我們新丟的資料也要正規化通通除以 255
x_Train_norm = X_train_2D/255
x_Test_norm = X_test_2D/255

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)

# 顯示訓練成果(分數)
scores = model.evaluate(x_Test_norm, y_TestOneHot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))


# 預測(prediction)
X = x_Test_norm[0:10,:]
predictions = model.predict_classes(X)
# get prediction result
print(predictions)

#可以手寫其它的圖片進去試試看，這邊有測試幾張，有些會不準確，發現它的訓練圖片幾乎都是黑色底白色字，
#所以我自己重新製作一張黑色底白色字的，它就預測正確但是如果不是黑底白字 預測錯的機率很高
img=cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
crop_size = (28, 28)
img = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)

# 顯示圖片
cv2.imshow('My Image', img)
# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

img_2D = img.reshape(1,28*28).astype('float32')
img_norm=img_2D/255
img = img_norm
predictions = model.predict_classes(img)
# get prediction result
print(predictions)