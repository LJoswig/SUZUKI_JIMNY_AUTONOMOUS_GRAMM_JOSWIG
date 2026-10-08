# `data_capture` package

Record datapoints while driving to create a big dataset.

---

## Data capturing
During driving this node will save datapoints that are `x` meters apart. Datapoints include:

- Left and Right ZED camera images
- Vehicle states
  - velocity
  - yaw rate
  - steering angle
  - pose
- Vehicle GPS position

The CSV file also contains additional vehicle state entries between different images taken.
This is indexed using the sub_count, which is reset everytime a new image is recorded.
The datapoint index starts at 0 and increases by one for every image recorded.
All data is stored in the recordings folder, where a new folder is created using the current timestamp `dd_mm_YYYY_HH_MM`.
Within this folder the CSV file is directly stored and the images are in a subfolder called `images`

## Post-processing
The idea is to automatically annotate the captured images using the future vehicle states recorded.
This could be done by first projecting the left and right camera image into a point cloud.
Next we could replay how the car has driven in the future to create the left and right tyre tracks.
The pixels in the point cloud that are near to the tyre tracks should be classified as driveable.
An object detection algorithm could also be ran to detect obstacles (such as cars, pedestrians or animals).
These detections could than also be annotated on the captured image. This last step does however pose the downside that 
the performance of the new model might be restricted by the performance of the used detection model.
This allows to easily and cheaply create a big dataset without much human interaction.
