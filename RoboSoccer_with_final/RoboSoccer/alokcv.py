import json
import cv2
import numpy as np
from cv2 import aruco
import requests 
import asyncio
import websockets
import threading
import time
import math
from filterpy.kalman import KalmanFilter
data_to_send='none'
result=[]


team1 = [6, 11]
team2 = [0, 1]


ARUCO_DICT = {
    1: cv2.aruco.DICT_4X4_50,
    2: cv2.aruco.DICT_4X4_100,
    3: cv2.aruco.DICT_4X4_250,
    4: cv2.aruco.DICT_4X4_1000,
    5: cv2.aruco.DICT_5X5_50,
    6: cv2.aruco.DICT_5X5_100,
    7: cv2.aruco.DICT_5X5_250,
    8: cv2.aruco.DICT_5X5_1000,
    9: cv2.aruco.DICT_6X6_50,
    10: cv2.aruco.DICT_6X6_100,
    11: cv2.aruco.DICT_6X6_250,
    12: cv2.aruco.DICT_6X6_1000,
    13: cv2.aruco.DICT_7X7_50,
    14: cv2.aruco.DICT_7X7_100,
    15: cv2.aruco.DICT_7X7_250,
    16: cv2.aruco.DICT_7X7_1000,
    17: cv2.aruco.DICT_ARUCO_ORIGINAL,
    18: cv2.aruco.DICT_APRILTAG_16h5,
    19: cv2.aruco.DICT_APRILTAG_25h9,
    20: cv2.aruco.DICT_APRILTAG_36h10,
    21: cv2.aruco.DICT_APRILTAG_36h11
}


selected_dict = 11
cap = cv2.VideoCapture(0)  #'http://192.168.137.185:4747/video'#100.76.82.17
marker_size=10
distance_ratio=0


camera_matrix = np.array([[640, 0, 320],
                            [0, 640, 240],
                            [0, 0, 1]], dtype=np.float32)
distortion_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

def find_square_center(marker_corners):
        
        marker_corners = np.array(marker_corners)
        avg_x = np.mean(marker_corners[:, :, 0])
        avg_y = np.mean(marker_corners[:, :, 1])

        return (round(avg_x), round(avg_y))


direction_vectors = []
extended_line_ends = []


#-->for angle of orientation of bot
def find_angle(corner1,corner2,centre_i):
        x=(corner1[0]+corner2[0])/2
        y=(corner1[1]+corner2[1])/2
        mid=(int(x),int(y))
        mid_vector=np.array(mid)-np.array(centre_i)
        # print("mid point is :",mid)
        angle=np.arctan2(mid_vector[1],mid_vector[0])
        angle_deg=np.degrees(angle)
        angle_deg=-1*angle_deg

        return angle_deg

def ball_tracking(frame):
    frame1 = frame.copy()
    frame1 = cv2.GaussianBlur(frame1, (7, 7), 0, 0)
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([15, 120, 120])
    upper_orange = np.array([18, 255, 255])

    ball_centre = None
    square = []

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res = cv2.bitwise_and(frame1, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            ball_centre = (x + w / 2, y + h / 2)
            square = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    frame1=frame
    frame1=cv2.GaussianBlur(frame1,(7,7),0,0)
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    
    lower_dark_blue = np.array([110, 20, 10])
    upper_dark_blue = np.array([130, 255, 255])
    blue_square = []



    blue_ball_centre = None
    mask = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
    
    
    res = cv2.bitwise_and(frame1, frame, mask=mask)
    
    # Find contours of the yellow objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            blue_ball_centre=(x+w/2,y+h/2)
            blue_square = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    
    
    frame1=frame
    frame1=cv2.GaussianBlur(frame1,(7,7),0,0)
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    red_square = []
    
    
    lower_red = np.array([0, 70, 10])
    upper_red = np.array([8, 255, 255])



    red_ball_centre = None
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    
    res = cv2.bitwise_and(frame1, frame, mask=mask)
    
    # Find contours of the yellow objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 400:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            red_ball_centre=(x+w/2,y+h/2)
            red_square = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    
    
   

    return frame, ball_centre, square,blue_ball_centre,blue_square,red_ball_centre,red_square



def marker_ROI(marker_corners,i):
        result_temp=[]
        center_x, center_y = find_square_center(marker_corners[i])
        center = (int(center_x), int(center_y))
       
        corner_point=((marker_corners[i][0][3][0]*1+marker_corners[i][0][0][0]*5)/6,(marker_corners[i][0][3][1]*1+marker_corners[i][0][0][1]*5)/6)
        corner_point1=((marker_corners[i][0][2][0]*1+marker_corners[i][0][1][0]*5)/6,(marker_corners[i][0][2][1]*1+marker_corners[i][0][1][1]*5)/6)
    
       
        direction_vector = np.array(corner_point) - np.array(center)
        direction_vector1 = np.array(corner_point1) - np.array(center)
        extended_line_end = (center[0] + direction_vector[0]*8, center[1] + direction_vector[1]*8 )
        extended_line_end1 = (center[0] + direction_vector1[0]*8, center[1] + direction_vector1[1]*8 )
     
        
        direction_vectors.append((direction_vector, direction_vector1))
        extended_line_ends.append((extended_line_end, extended_line_end1))
        angle=find_angle(corner_point,corner_point1,center)
        angle=round(angle,2)

    
        return angle,center,extended_line_end,extended_line_end1,result



def line_intersects_square(line_start, line_end, square_corners):
    x1, y1= line_start
    x2, y2 = line_end
    for i in range(4):
        x3, y3 = square_corners[i]
        x4, y4 = square_corners[(i + 1) % 4]

        if line_intersects_line(x1, y1, x2, y2, x3, y3, x4, y4):
            return True

    return False

def line_intersects_line(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(x1, y1, x2, y2, x3, y3):
        return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)

    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and \
           ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)

def distance(point1,point2):
     return math.sqrt((point1[1]-point2[1])**2+(point1[0]-point2[0])**2)

def draw_square(image, corners):
    # Draw each side of the square
    for i in range(4):
        cv2.line(image, corners[i], corners[(i + 1) % 4], (255, 0, 0), 2)


def adjust_end_point(frame, start_point, end_point):
    frame_height, frame_width = frame.shape[:2]
    x1, y1 = start_point
    x2, y2 = end_point

    intersections = []

    if x1 != x2:
        left_y = y1 + ((0 - x1) / (x2 - x1)) * (y2 - y1)
        if 0 <= left_y <= frame_height:
            intersections.append((0, int(left_y)))

   
    if x1 != x2:
        right_y = y1 + ((frame_width - x1) / (x2 - x1)) * (y2 - y1)
        if 0 <= right_y <= frame_height:
            intersections.append((frame_width, int(right_y)))

    if y1 != y2:
        top_x = x1 + ((0 - y1) / (y2 - y1)) * (x2 - x1)
        if 0 <= top_x <= frame_width:
            intersections.append((int(top_x), 0))

    if y1 != y2:
        bottom_x = x1 + ((frame_height - y1) / (y2 - y1)) * (x2 - x1)
        if 0 <= bottom_x <= frame_width:
            intersections.append((int(bottom_x), frame_height))

    distances = [((end_point[0] - inter[0])**2 + (end_point[1] - inter[1])**2)**0.5 for inter in intersections]

    closest_index = np.argmin(distances)
    if closest_index is not None:
        end_point = intersections[closest_index]
   
    intersection_x, intersection_y = intersections[closest_index]
    
    

    cv2.circle(frame, end_point, 5, (0, 255, 0), -1)
    
    opposite_side_intersection=find_opposite_point(frame,start_point,end_point)
    # cv2.circle(frame, opposite_side_intersection, 5, (0, 255, 255), -1)

    return end_point,opposite_side_intersection

def find_opposite_point(frame, start_point, end_point):
    frame_height, frame_width = frame.shape[:2]

    # Calculate line slope and intercept (handle vertical lines)
    if end_point[0] != start_point[0]:
        slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        intercept = start_point[1] - slope * start_point[0]
    else:
        slope = None  # Vertical line
        intercept = start_point[0]  # Any x-value on the line

    # Calculate the opposite point based on endpoint position
    if slope is not None:  # Non-vertical line
        if end_point[0] == 0:
            opposite_x = frame_width
            opposite_y = slope * opposite_x + intercept
        elif end_point[1] == 0:
            opposite_y = frame_height
            opposite_x = (opposite_y - intercept) / slope
        elif end_point[0] == frame_width:
            opposite_x = 0
            opposite_y = slope * opposite_x + intercept
        elif end_point[1] == frame_height:
            opposite_y = 0
            opposite_x = (opposite_y - intercept) / slope
    else:  # Vertical line
        if end_point[0] == 0:
            opposite_x = frame_width
            opposite_y = start_point[1]  # Same y-value as start_point
        elif end_point[1] == 0:
            opposite_y = frame_height
            opposite_x = start_point[0]  # Same x-value as start_point
        elif end_point[0] == frame_width:
            opposite_x = 0
            opposite_y = start_point[1]  # Same y-value as start_point
        elif end_point[1] == frame_height:
            opposite_y = 0
            opposite_x = start_point[0]  # Same x-value as start_point

    # Clamp coordinates within frame boundaries
    opposite_x = max(0, min(opposite_x, frame_width - 1))
    opposite_y = max(0, min(opposite_y, frame_height - 1))

    return int(opposite_x), int(opposite_y)
def blue_goal_post(frame):
    frame1=frame
    frame1=cv2.GaussianBlur(frame1,(7,7),0,0)
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    
    
    lower_dark_blue = np.array([110, 100, 100])
    upper_dark_blue = np.array([130, 255, 255])
    square = []



    ball_centre = None
    mask = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
    
    
    res = cv2.bitwise_and(frame1, frame, mask=mask)
    
    # Find contours of the yellow objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ball_centre=(x+w/2,y+h/2)
            square = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    
    
    return frame,ball_centre,square
def red_goal_post(frame):
    frame1=frame
    frame1=cv2.GaussianBlur(frame1,(7,7),0,0)
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    square = []
    
    
    lower_red = np.array([5, 70, 10])
    upper_red = np.array([10, 255, 255])



    ball_centre = None
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    
    res = cv2.bitwise_and(frame1, frame, mask=mask)
    
    # Find contours of the yellow objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 400:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ball_centre=(x+w/2,y+h/2)
            square = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    
    
    return frame,ball_centre,square

def homography_transform(frame,calliberation_coordinate):
    im_src=frame
    im_dst=frame
 
    pts_src = np.array(calliberation_coordinate)
    h,w,_=frame.shape
    
    pts_dst = np.array([[0, 0],[w, 0],[w, h],[0, h]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    return im_out



def apply_wiener_filter(frame, kernel_size=(3, 3)):
   
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    filtered_frame = cv2.filter2D(gray_frame, -1, np.ones(kernel_size, dtype=np.float32)/(kernel_size[0]*kernel_size[1]))
    
    return filtered_frame

def main_function():
    
    print("main function started")
    while True:
        ret, frame = cap.read()
        # frame = apply_wiener_filter(frame)
        frame=homography_transform(frame,[[136, 15], [599, 157], [592, 479], [1, 322]])
        
        if not ret:
            break
        

        _, ball_centre, circle_square,blue_goalPost_centre,b_goalPost_square,red_goalPost_centre,r_goalPost_square= ball_tracking(frame)
        # _,ball_centre,circle_square = ball_tracking(frame)
        # _,blue_goalPost_centre,b_goalPost_square = blue_goal_post(frame)
        # _,red_goalPost_centre,r_goalPost_square = red_goal_post(frame)

        # Detection of markers
        marker_dict = aruco.getPredefinedDictionary(ARUCO_DICT[selected_dict])
        param_marker = aruco.DetectorParameters()

        param_marker.adaptiveThreshWinSizeMin = 3
        param_marker.adaptiveThreshWinSizeMax = 20
        param_marker.adaptiveThreshWinSizeStep = 10
        param_marker.adaptiveThreshConstant = 40
        param_marker.minMarkerPerimeterRate = 0.003
        param_marker.maxMarkerPerimeterRate = 4.0
        param_marker.polygonalApproxAccuracyRate = 0.03
        param_marker.minCornerDistanceRate = 0.05
        param_marker.minDistanceToBorder = 3
        param_marker.minMarkerDistanceRate = 0.05
        param_marker.cornerRefinementWinSize = 5
        param_marker.cornerRefinementMaxIterations = 5
        param_marker.cornerRefinementMinAccuracy = 0.1
        param_marker.markerBorderBits = 1
        param_marker.perspectiveRemovePixelPerCell = 4
        param_marker.perspectiveRemoveIgnoredMarginPerCell = 0.13
        param_marker.maxErroneousBitsInBorderRate = 0.35
        param_marker.minOtsuStdDev = 14.0
        param_marker.errorCorrectionRate = 0.8
        # Set parameters
        
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _,gray_img=cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
        
        detector = cv2.aruco.ArucoDetector(marker_dict, param_marker)
        
        marker_corners, marker_ids, _ = detector.detectMarkers(gray_img)
        

        if marker_corners:
            

            # for i in range(len(marker_ids)):
            
            #     object_points = np.array([[0, 0, 0], 
            #                             [marker_size, 0, 0], 
            #                             [marker_size, marker_size, 0], 
            #                             [0, marker_size, 0]], dtype=np.float32)

            #     _, rvec, tvec = cv2.solvePnP(object_points, marker_corners[i], camera_matrix, distortion_coeffs)
                
            #     axis_points, _ = cv2.projectPoints(np.float32([[0,0,0], [10,0,0], [0,10,0], [0,0,10]]), 
            #                                         rvec, tvec, camera_matrix, distortion_coeffs)
                
            #     # frame = cv2.line(frame, tuple(marker_corners[i][0][0].astype(int)), tuple(axis_points[1][0].astype(int)), (0, 0, 255), 5)
            #     # frame = cv2.line(frame, tuple(axis_points[2][0].astype(int)), tuple(marker_corners[i][0][0].astype(int)), (0, 255, 0), 5)
            #     # frame = cv2.line(frame, tuple(marker_corners[i][0][0].astype(int)), tuple(axis_points[3][0].astype(int)), (255, 0, 0), 5)
            
        
            # for ids, corners in zip(marker_ids, marker_corners):
            #     aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            #     # cv2.polylines(frame, [corners.astype(np.int32)], True, (0,255,255), 1, cv2.LINE_AA)

            for i in range(len(marker_ids)):
                # print(f"marker ids:{marker_ids[i]}")
                current_team, opponent_team = team1, team2
                if marker_ids[i] in team2:
                    current_team, opponent_team = team2, team1
            
                 
                angle, centre, extended_line_end, extended_line_end1, result = marker_ROI(marker_corners, i)

                scale_factor = 10
                extended_line_end = (centre[0] + scale_factor * (extended_line_end[0] - centre[0]), 
                            centre[1] + scale_factor * (extended_line_end[1] - centre[1]))

                extended_line_end1 = (centre[0] + scale_factor * (extended_line_end1[0] - centre[0]), 
                                    centre[1] + scale_factor * (extended_line_end1[1] - centre[1]))
                extended_corner2 = (centre[0] + scale_factor * (marker_corners[i][0][2][0] - centre[0]), 
                        centre[1] + scale_factor * (marker_corners[i][0][2][1] - centre[1]))

                extended_corner3 = (centre[0] + scale_factor * (marker_corners[i][0][3][0] - centre[0]), 
                                    centre[1] + scale_factor * (marker_corners[i][0][3][1] - centre[1]))

                endpoints = []
                endpoints.append(extended_line_end)
                for z in range(1, 10):
                    ratio = z / 10
                    new_point_x = extended_line_end[0] * (1 - ratio) + extended_line_end1[0] * ratio
                    new_point_y = extended_line_end[1] * (1 - ratio) + extended_line_end1[1] * ratio
                    endpoints.append((new_point_x, new_point_y))

                endpoints.append(extended_line_end1)
                endpoints.append(extended_corner2)
                endpoints.append((((extended_corner3[0]+extended_corner2[0])/2),(extended_corner3[1]+extended_corner2[1])/2))
                endpoints.append(extended_corner3)
                
                adjusted_endpoints = []


                for rayid, endpoint in enumerate(endpoints):
                    RayObv=[0]*8
                    adjusted_endpoint, opposite_endpoint = adjust_end_point(frame, centre, endpoint)
                    boundary_points=adjusted_endpoint
                    total_ray_distance=distance(adjusted_endpoint,opposite_endpoint)
                    half_distance=distance(adjusted_endpoint,centre)
                    
                    min_distance_ratio=half_distance/total_ray_distance

                    hitTag = 3
                    # mindistance = 10000

                    for l in range(len(marker_ids)):
                        if marker_ids[i] != marker_ids[l]:
                            center_x_j, center_y_j = find_square_center(marker_corners[l])

                            square=(marker_corners[l][0][0], marker_corners[l][0][1], marker_corners[l][0][2], marker_corners[l][0][3])
                            square=[s.astype(int).tolist() for s in square]
                            currhitTag=3
                            if line_intersects_square(centre,adjusted_endpoint,square):
                                if marker_ids[l] in current_team:
                                    currhitTag = 4
                                elif marker_ids[l] in opponent_team:
                                    currhitTag = 5

                                adjusted_endpoint=(center_x_j,center_y_j)
                                distance_intersection_point=distance(adjusted_endpoint,centre)
                                distance_ratio=distance_intersection_point/total_ray_distance
                                if distance_ratio < min_distance_ratio:
                                    min_distance_ratio = distance_ratio
                                    hitTag = currhitTag

                    
                    if len(circle_square)>=4 and line_intersects_square(centre,adjusted_endpoint,circle_square):
                        adjusted_endpoint=ball_centre
                        distance_intersection_point=distance(adjusted_endpoint,centre)
                        distance_ratio=distance_intersection_point/total_ray_distance
                        if distance_ratio < min_distance_ratio:
                            min_distance_ratio = distance_ratio
                            hitTag = 0
                    if len(b_goalPost_square)>=4 and line_intersects_square(centre,adjusted_endpoint,b_goalPost_square):
                        adjusted_endpoint=blue_goalPost_centre
                        distance_intersection_point=distance(adjusted_endpoint,centre)
                        distance_ratio=distance_intersection_point/total_ray_distance
                        if distance_ratio < min_distance_ratio:
                            min_distance_ratio = distance_ratio
                            hitTag = 1
                    if len(r_goalPost_square)>=4 and line_intersects_square(centre,adjusted_endpoint,r_goalPost_square):
                        adjusted_endpoint=red_goalPost_centre
                        distance_intersection_point=distance(adjusted_endpoint,centre)
                        distance_ratio=distance_intersection_point/total_ray_distance
                        if distance_ratio < min_distance_ratio:
                            min_distance_ratio = distance_ratio
                            hitTag = 2

                    RayObv[7]=round(min_distance_ratio, 2)
                    RayObv[hitTag] = 1
                    adjusted_endpoints.append(adjusted_endpoint)
                    # if adjusted_endpoint==boundary_points:
                    #     RayObv[3]=1
                    print(f"for agent{i} ray{rayid} 1 hot bit-->",RayObv)


                for endpoint in adjusted_endpoints:
                    cv2.arrowedLine(frame, (int(centre[0]), int(centre[1])), (int(endpoint[0]), int(endpoint[1])), (255, 0, 255), 2)
                print("="*20)


        frame,ball_centre,_,_,_,_,_=ball_tracking(frame)
        cv2.putText(frame,f"ball centre:{str(ball_centre)} ",(400,50), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),2)
        
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

