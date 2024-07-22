import numpy as np
import cv2
import Person 
import serial
import struct
import socket
import time
import threading


carno = 0
cut_in = 0
cut_out = 0

def print_hello():

  global cnt_in
  cnt_in=0
  global cut_out
  cnt_out=0
  count_in=0
  count_out=0
  state=0

  font = cv2.FONT_HERSHEY_SIMPLEX
  persons = []
  rect_co = []
  max_p_age = 1
  pid = 1
  val = []

  video=cv2.VideoCapture("counting_test.avi")
  #输出视频
  fourcc = cv2.VideoWriter_fourcc(*'XVID')#输出视频制编码

  w = video.get(3)
  h = video.get(4)
  area = h*w
  areaTHreshold = area/500

  #计算画线的位置
  line_up = int(1*(h/4))
  line_down = int(2.7*(h/4))
  up_limit = int(.5*(h/4))
  down_limit = int(3.2*(h/4))

  line_down_color = (255,0,0)
  line_up_color = (0,255,0)
  pt1 = [0, line_down]
  pt2 = [w, line_down]
  pts_L1 = np.array([pt1,pt2], np.int32)
  pts_L1 = pts_L1.reshape((-1,1,2))
  pt3 = [0, line_up]
  pt4 = [w, line_up]
  pts_L2 = np.array([pt3,pt4], np.int32)
  pts_L2 = pts_L2.reshape((-1,1,2))

  pt5 = [0, up_limit]
  pt6 = [w, up_limit]
  pts_L3 = np.array([pt5,pt6], np.int32)
  pts_L3 = pts_L3.reshape((-1,1,2))
  pt7 =  [0, down_limit]
  pt8 =  [w, down_limit]
  pts_L4 = np.array([pt7,pt8], np.int32)
  pts_L4 = pts_L4.reshape((-1,1,2))
  #背景剔除
  fgbg = cv2.createBackgroundSubtractorKNN()
  #用于后面形态学处理的核
  kernel = np.ones((3,3),np.uint8)
  kerne2 = np.ones((5,5),np.uint8)
  kerne3 = np.ones((11,11),np.uint8)

  # 添加初始化总人数计数器
  total_people_count = 0

  while(video.isOpened()):
      ret,frame=video.read()
      if frame is None:
          break

      # 每次处理新帧时重置当前帧人数计数器
      current_frame_people_count = 0

      #应用背景剔除
      gray = cv2.GaussianBlur(frame, (31, 31), 0)
      fgmask = fgbg.apply(gray)
      fgmask2 = fgbg.apply(gray)

      try:
          #二值化
          ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
          ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
          #开操作(腐蚀->膨胀)消除噪声
          mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kerne3)
          mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kerne3)
          #闭操作(膨胀->腐蚀)将区域连接起来
          mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kerne3)
          mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kerne3)

      except:
          print('EOF')
          print ('IN:',cnt_in+count_in)
          print ('OUT:',cnt_in+count_in)
          break

      #找到边界
      contours0, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours0:
          rect = cv2.boundingRect(cnt)#矩形边框
          area=cv2.contourArea(cnt)#每个矩形框的面积
          if area>areaTHreshold:

              # 对于每个检测到的人，增加当前帧的人数计数器
              current_frame_people_count += 1

              #moments里包含了许多有用的信息
              M=cv2.moments(cnt)
              cx=int(M['m10']/M['m00'])#计算重心
              cy=int(M['m01']/M['m00'])
              x, y, w, h = cv2.boundingRect(cnt)#x,y为矩形框左上方点的坐标，w为宽，h为高
              new=True
              if cy in range(up_limit,down_limit):
                  for i in persons:
                      if abs(cx-i.getX())<=w and abs(cy-i.getY())<=h:
                          new=False
                          i.updateCoords(cx,cy)
                          if i.going_UP(line_down,line_up)==True:
                              if w>80:
                                  count_in=w/40
                                  #print("In:执行了/60")
                              else:
                                  cnt_in+=1
                          elif i.going_DOWN(line_down,line_up)==True:
                              if w>80:
                                  count_out=w/40
                              else:
                                  cnt_out+=1
                          break
                          #状态为1表明
                      if i.getState() == '1':
                          if i.getDir() == 'down' and i.getY() > down_limit:
                              i.setDone()
                          elif i.getDir() == 'up' and i.getY() < up_limit:
                              i.setDone()
                      if i.timedOut():
                          # 已经记过数且超出边界将其移出persons队列
                          index = persons.index(i)
                          persons.pop(index)
                          del i  # 清楚内存中的第i个人
                  if new == True:
                      p = Person.MyPerson(pid, cx, cy, max_p_age)
                      persons.append(p)
                      pid += 1
              #矩形框加中心原点标记行人
              cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
              img = cv2.rectangle(frame, (x, y), (x + w, y + h),line_up_color, 2)
      for i in persons:
          aaa=1
      frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
      frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
      frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
      frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

      # 累加到总人数计数器
      #total_people_count += current_frame_people_count
      global str_current_frame
      str_current_frame = 'People Count:' + str(cnt_in + cnt_out)
      cv2.putText(frame, str_current_frame, (240, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
      cv2.putText(frame, str_current_frame, (240, 70), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

      cv2.namedWindow("people_counter", 0)
      cv2.resizeWindow("people_counter", 650, 480)
      cv2.moveWindow("people_counter",100,100)
      cv2.imshow('people_counter', frame)

      k = cv2.waitKey(20) & 0xff
      if k == 27:
          break
  video.release()
  cv2.destroyAllWindows()


"""------------------------------------------------------------------------------------------------------------------"""


def print_numbers(num):

    min_w = 90
    min_h = 90

    # 检测线的高度
    line_high = 550
    # 线的偏移
    offset = 7

    # 统计车的数量
    global carno
    carno = 0
    # 存放有效车辆的数组
    cars = []

    def center(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1

        return cx, cy
    '''
    # 创建UDP套接字
    UDP_IP = "192.168.0.103"  # 使用您的服务器IP更新
    UDP_PORT = 8081  # 使用端口号更新
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    bgsubmog = cv2.createBackgroundSubtractorMOG2()
    # 形态学kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while True:
        # 接收帧数据
        data, addr = sock.recvfrom(65507)  # 根据需要调整缓冲区大小
        frame = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(frame, flags=1)

        # 检查是否收到帧
        if frame is not None:
            # 灰度
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 去噪（高斯）
            blur = cv2.GaussianBlur(frame, (3, 3), 5)
            # 去背影
            mask = bgsubmog.apply(blur)

            # 腐蚀，去掉图中小斑块
            erode = cv2.erode(mask, kernel)

            # 膨胀，还原放大
            dilate = cv2.dilate(erode, kernel, iterations=3)

            # 闭操作，去掉物体内部的小块
            close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
            close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)

            cnts, h = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 画一条检测线
            cv2.line(frame, (10, line_high), (1200, line_high), (255, 255, 0), 3)

            for (i, c) in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(c)

                # 对车辆的宽高进行判断以验证是否是有效的车辆
                isValid = (w >= min_w) and (h >= min_h)
                if not isValid:
                    continue

                # 到这里都是有效的车
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cpoint = center(x, y, w, h)
                cars.append(cpoint)
                cv2.circle(frame, (cpoint), 5, (0, 0, 255), -1)

                for (x, y) in cars:
                    if (y > line_high - offset) and (y < line_high + offset):
                        carno += 1
                        cars.remove((x, y))
                        print(carno)

            cv2.putText(frame, "Cars Count:" + str(carno), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            cv2.imshow('frame', frame)
            

        key = cv2.waitKey(1)
        if key == 27:
            break

    # 关闭UDP套接字
    sock.close()
    cv2.destroyAllWindows()
'''
    cap = cv2.VideoCapture('video.mp4')

    bgsubmog = cv2.createBackgroundSubtractorMOG2()
    # 形态学kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while True:
        ret, frame = cap.read()
        if (ret == True):

            # 灰度
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 去噪（高斯）
            blur = cv2.GaussianBlur(frame, (3, 3), 5)
            # 去背影
            mask = bgsubmog.apply(blur)
            # 腐蚀， 去掉图中小斑块
            erode = cv2.erode(mask, kernel)
            # 膨胀， 还原放大
            dilate = cv2.dilate(erode, kernel, iterations=3)
            # 闭操作，去掉物体内部的小块
            close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
            close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)

            cnts, h = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 画一条检测线
            cv2.line(frame, (10, line_high), (1200, line_high), (255, 255, 0), 3)

            for (i, c) in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(c)

                # 对车辆的宽高进行判断
                # 以验证是否是有效的车辆
                isValid = (w >= min_w) and (h >= min_h)
                if (not isValid):
                    continue

                # 到这里都是有效的车
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cpoint = center(x, y, w, h)
                cars.append(cpoint)
                cv2.circle(frame, (cpoint), 5, (0, 0, 255), -1)

                for (x, y) in cars:
                    if ((y > line_high - offset) and (y < line_high + offset)):
                        carno += 1
                        cars.remove((x, y))
                        # print(carno)

            cv2.putText(frame, "Cars Count:" + str(carno), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            global carNumber

            cv2.namedWindow("cars_counter", 0)
            cv2.resizeWindow("cars_counter", 650, 480)
            cv2.moveWindow("cars_counter", 700, 100)
            cv2.imshow('cars_counter', frame)

        key = cv2.waitKey(1)
        if (key == 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def test2():
    while 1 == 1:
        global str_current_frame
        global carNumber
        time.sleep(2)

thread1 = threading.Thread(target=print_hello, args=())
thread2 = threading.Thread(target=print_numbers, args=(10,))
thread3 = threading.Thread(target=test2, args=())

thread1.start()
thread2.start()
thread3.start()


"""------------------------------------------------------------------------------------------------------------------"""


def calculate_crc(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    crc_bytes = struct.pack('<H', crc)
    return crc_bytes

def send_with_crc(ser, data):
    crc = calculate_crc(data)
    data_with_crc = data + crc
    ser.write(data_with_crc)

def main():
    sec1 = 5
    sec2 = 20
    while True:

        print("绿灯常亮")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 02 FF 00 2D FA"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 00 FF 00 8C 3A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        time.sleep(1)
        for i in range(sec1, 0, -1):
            print(i)
            time.sleep(1)

        print("绿灯不亮")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 02 00 00 6C 0A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 00 00 00 CD CA"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        print("黄灯慢闪")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 09 FF 00 5C 38"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 09 FF 00 5C 38"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        for i in range(5, 0, -1):
            print(i)
            time.sleep(1)

        print("黄灯不亮")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 01 00 00 9C 0A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 01 00 00 9C 0A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        print("红灯常亮")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 00 FF 00 8C 3A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 02 FF 00 2D FA"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        for i in range(sec2, 0, -1):
            print(i)
            time.sleep(1)

        print("红灯不亮")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 00 00 00 CD CA"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 02 00 00 6C 0A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        print("黄灯慢闪")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 09 FF 00 5C 38"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 09 FF 00 5C 38"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        for i in range(5, 0, -1):
            print(i)
            time.sleep(1)

        print("黄灯不亮")
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "01 05 00 01 00 00 9C 0A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()
        ser = serial.Serial('COM5', 9600, timeout=1)
        input_string = "05 05 00 01 00 00 9C 0A"
        try:
            data = bytes.fromhex(input_string)
        except ValueError:
            print("Invalid hex string format.")
            continue
        send_with_crc(ser, data)
        response = ser.read(1)
        ser.close()

        if (cnt_in + cnt_out) > 20:
            sec1 -= 5
        elif carno/(cnt_in + cnt_out) > 6:
            sec1 += 5
        sec2 = 60 - sec1

        cnt_in = 0
        cnt_out = 0
        carno = 0

if __name__ == "__main__":
    main()