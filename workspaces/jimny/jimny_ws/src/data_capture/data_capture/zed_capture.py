#!/usr/bin/python3

"""
zed_capture.py
Author: Matthijs Steyerberg, Axel Barbelanne
Date: 22-10-2024

Open a ZED camera and record an image to a specified directory.
The preferred method however is to use the ROS2 ZED Node instead.
This other method allows to use the cameras in multiple instances.
"""
import pyzed.sl as sl
import cv2
import os


class ZEDCAMERA():
    def __init__(self, folder='images'):
        # Create a Camera object
        self.zed = sl.Camera()

        # Create an InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.AUTO  # Use HD720 or HD1200 video mode, depending on camera type.
        self.init_params.camera_fps = 30  # Set fps at 30

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open  Failed: " + repr(err) + ". Exiting program.")
            exit()

        # Create the images directory if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        self.runtime_parameters = sl.RuntimeParameters()
    
    def capture_img(self, i=0):
        image = sl.Mat()
        # Grab an image, a RuntimeParameters object must be given to grab()
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
            
            # Convert the image to a format suitable for OpenCV
            image_cv = image.get_data()
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)  # Convert RGBA to BGR

            # Save the image in the 'images' subfolder
            filename = f'images/image_{i:03d}_{timestamp.get_milliseconds()}.jpg'
            cv2.imwrite(filename, image_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 85])  # Set quality to 85 (adjustable)

            print(f"Saved {filename} | Image resolution: {image.get_width()} x {image.get_height()} || Image timestamp: {timestamp.get_milliseconds()}")

    def __del__(self):
        # Close the camera
        self.zed.close()

if __name__ == "__main__":
    zed = ZEDCAMERA()
    zed.capture_img()
    del zed

