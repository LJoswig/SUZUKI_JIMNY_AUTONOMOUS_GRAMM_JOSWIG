#Function: Use ZED camera depth sensing to capture and display a 3D point cloud.

#Libraries
import sys
import viewer as gl
import pyzed.sl as sl #Import the ZED SDK Python API
import argparse

#define function: selects the image source and resolution 
def parse_args(init, opt):
    #check whether SVO recording or network stream
    #in Terminal: python3 depth_sensing.py --input_svo_file testfahrt.svo2
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else : #If no SVO file or no IP address use the live images of connected ZED X camera
            print("Unvalid IP format. Using live stream of connected camera")
    #check whether resolution is specified in Terminal
    #in Terminal: python3 depth_sensing.py --resolution HD1200         
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    #default resolution is HD1200: 1920 × 1200 pixels per image
    else : 
        print("[Sample] Using default resolution")

#main function:
def main(opt):
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud\nPress 'v' to toggle voxel display")

    use_gpu = gl.GPU_ACCELERATION_AVAILABLE and not opt.disable_gpu_data_transfer
    mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU
    if use_gpu:
        print("🚀 Using GPU data transfer with CuPy")
    #create camera settings object init
    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.NEURAL,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    #call the function parse_args
    parse_args(init, opt)
    #open the camera using the settings stored in init
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = -1
    res.height = -1

    point_cloud = sl.Mat()
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, res)
    res = point_cloud.get_resolution()

    #initiliase the OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, res)

    #Keep capturing and displaying point clouds while the viewer is open
    while viewer.is_available():
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            # Retrieve point cloud data: voxel-decimated or full resolution
            if viewer.use_voxels:
                voxel_params = sl.VoxelMeasureParameters()
                voxel_params.voxel_size = 0.025  # 25 mm expressed in meters (coordinate_units = METER)
                zed.retrieve_voxel_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, voxel_params)
            else:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, res)
            viewer.updateData(point_cloud)
            if viewer.save_data:
                # For saving, we take CPU memory regardless of processing type
                point_cloud_to_save = sl.Mat()
                zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                err = point_cloud_to_save.write('Pointcloud.ply')
                if(err <= sl.ERROR_CODE.SUCCESS):
                    print("Current .ply file saving succeed")
                else:
                    print("Current .ply file failed")
                viewer.save_data = False
    viewer.exit()
    zed.close()

#Read command line arguments and start the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--disable-gpu-data-transfer', action='store_true', help='Disable GPU data transfer acceleration with CuPy even if CuPy is available')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt) #start the main function
