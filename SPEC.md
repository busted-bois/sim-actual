The AI controller has access to:

- Forward camera images (30 Hz)
- Vehicle attitude
- Angular rates
- Accelerometer measurements
- Gyroscope measurements
- Magnetometer measurements
- Pressure measurements
- Temperature measurements
- Heartbeat status
- Timing synchronization

The AI controller does NOT have access to:

- GPS
- Absolute global position

# Communication Protocol

The simulator communicates using MAVLink.  
Contestant code should communicate with the drone using MAVSDK-compatible libraries.

**Some methods from [https://github.com/mavlink/c_library_v2](https://github.com/mavlink/c_library_v2) are valid**

These 6 messages were in the specification pdf provided by the competition.  
Other messages from MAVLink 2’s protocol should be supported as documented in [https://mavlink.io/en/messages/common.html](https://mavlink.io/en/messages/common.html)

(Note: Very good mavlink documentation, please look at it if the 6 messages do not have input/output you need)

MAVLink output message from the **simulator to client:**  
**1\. HEARTBEAT (MAVLink ID 0\)**  
The heartbeat message shows that a system or component is present and responding. The type and autopilot fields (along with the message component id), allow the receiving system to treat further messages from this system appropriately (e.g. by laying out the user interface based on the autopilot). [https://mavlink.io/en/services/heartbeat.html](https://mavlink.io/en/services/heartbeat.html)

**2\. ATTITUDE (MAVLink ID 30\)**  
[https://mavlink.io/en/guide/mavlink_2.html](https://mavlink.io/en/guide/mavlink_2.html)  
Output vehicle attitude information. More specifically outputs the attitude in the aeronautical frame (right-handed, Z-down, Y-right, X-front, ZYX, intrinsic).

| Field Name   | Type     | Units | Description                         |
| :----------- | :------- | :---- | :---------------------------------- |
| time_boot_ms | uint32_t | ms    | Timestamp (time since system boot). |
| roll         | float    | rad   | Roll angle (-pi..+pi)               |
| pitch        | float    | rad   | Pitch angle (-pi..+pi)              |
| yaw          | float    | rad   | Yaw angle (-pi..+pi)                |
| rollspeed    | float    | rad/s | Roll angular speed                  |
| pitchspeed   | float    | rad/s | Pitch angular speed                 |
| yawspeed     | float    | rad/s | Yaw angular speed                   |

**3\. HIGHRES_IMU (MAVLink ID 105\)**

The IMU readings in SI units in NED body frame

IMU stands for Inertial Measurement Unit which contains the sensors for Accelerometer, Gyroscope, and Magnetometer which measures the acceleration (movement and tilt), measures rotation/angular velocity, and measures direction like a compass respectively.

| Field Name                                                                | Type     | Units | Values                                                                                            | Description                                                                                                                                                                                                          |
| :------------------------------------------------------------------------ | :------- | :---- | :------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| time_usec                                                                 | uint64_t | us    |                                                                                                   | Timestamp                                                                                                                                                                                                            |
| xacc                                                                      | float    | m/s/s |                                                                                                   | X acceleration                                                                                                                                                                                                       |
| yacc                                                                      | float    | m/s/s |                                                                                                   | Y acceleration                                                                                                                                                                                                       |
| zacc                                                                      | float    | m/s/s |                                                                                                   | Z acceleration                                                                                                                                                                                                       |
| xgyro                                                                     | float    | rad/s |                                                                                                   | Angular speed around X axis                                                                                                                                                                                          |
| ygyro                                                                     | float    | rad/s |                                                                                                   | Angular speed around Y axis                                                                                                                                                                                          |
| zgyro                                                                     | float    | rad/s |                                                                                                   | Angular speed around Z axis                                                                                                                                                                                          |
| xmag                                                                      | float    | gauss |                                                                                                   | X Magnetic field                                                                                                                                                                                                     |
| ymag                                                                      | float    | gauss |                                                                                                   | Y Magnetic field                                                                                                                                                                                                     |
| zmag                                                                      | float    | gauss |                                                                                                   | Z Magnetic field                                                                                                                                                                                                     |
| abs_pressure                                                              | float    | hPa   |                                                                                                   | Absolute pressure                                                                                                                                                                                                    |
| diff_pressure                                                             | float    | hPa   |                                                                                                   | Differential pressure                                                                                                                                                                                                |
| pressure_alt                                                              | float    |       |                                                                                                   | Altitude calculated from pressure                                                                                                                                                                                    |
| temperature                                                               | float    | degC  |                                                                                                   | Temperature                                                                                                                                                                                                          |
| fields_updated                                                            | uint16_t |       | [HIGHRES_IMU_UPDATED_FLAGS](https://mavlink.io/en/messages/common.html#HIGHRES_IMU_UPDATED_FLAGS) | Bitmap for fields that have updated since last message                                                                                                                                                               |
| id [\++](https://mavlink.io/en/messages/common.html#mav2_extension_field) | uint8_t  |       |                                                                                                   | Id. Ids are numbered from 0 and map to IMUs numbered from 1 (e.g. IMU1 will have a message with id=0) \[Instance field\]: Uniquely identifies a device/subcomponent within a single source/target MAVLink component. |

**4\. TIMESYNC (Mavlink id 111\)**  
Time synchronization message. The message is used for both timesync requests and responses. The request is sent with ts1=syncing component timestamp and tc1=0, and may be broadcast or targeted to a specific system/component. The response is sent with ts1=syncing component timestamp (mirror back unchanged), and tc1=responding component timestamp, with the target_system and target_component set to ids of the original request. Systems can determine if they are receiving a request or response based on the value of tc. If the response has target_system==target_component==0 the remote system has not been updated to use the component IDs and cannot reliably timesync; the requester may report an error. Timestamps are UNIX Epoch time or time since system boot in nanoseconds (the timestamp format can be inferred by checking for the magnitude of the number; generally it doesn't matter as only the offset is used). The message sequence is repeated numerous times with results being filtered/averaged to estimate the offset. See also: [https://mavlink.io/en/services/timesync.html](https://mavlink.io/en/services/timesync.html).

| Field Name                                                                              | Type    | Units | Description                                                                                                                              |
| :-------------------------------------------------------------------------------------- | :------ | :---- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| tc1                                                                                     | int64_t | ns    | Time sync timestamp 1\. Syncing: 0\. Responding: Timestamp of responding component.                                                      |
| ts1                                                                                     | int64_t | ns    | Time sync timestamp 2\. Timestamp of syncing component (mirrored in response).                                                           |
| target_system [\++](https://mavlink.io/en/messages/common.html#mav2_extension_field)    | uint8_t |       | Target system id. Request: 0 (broadcast) or id of specific system. Response must contain system id of the requesting component.          |
| target_component [\++](https://mavlink.io/en/messages/common.html#mav2_extension_field) | uint8_t |       | Target component id. Request: 0 (broadcast) or id of specific component. Response must contain component id of the requesting component. |

MAVLink output message from the **client to simulator:**  
**5\. SET_POSITION_TARGET_LOCAL_NED (84)**

Sets a desired vehicle position in a local north-east-down coordinate frame. Used by an external controller to command the vehicle (manual controller or other system).

| Field Name       | Type     | Units | Values                                                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                               |
| :--------------- | :------- | :---- | :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| time_boot_ms     | uint32_t | ms    |                                                                                                 | Timestamp (time since system boot).                                                                                                                                                                                                                                                                                                                                                                                       |
| target_system    | uint8_t  |       |                                                                                                 | System ID                                                                                                                                                                                                                                                                                                                                                                                                                 |
| target_component | uint8_t  |       |                                                                                                 | Component ID                                                                                                                                                                                                                                                                                                                                                                                                              |
| coordinate_frame | uint8_t  |       | [MAV_FRAME](https://mavlink.io/en/messages/common.html#MAV_FRAME)                               | Valid options are: [MAV_FRAME_LOCAL_NED](https://mavlink.io/en/messages/common.html#MAV_FRAME_LOCAL_NED) \= 1, [MAV_FRAME_LOCAL_OFFSET_NED](https://mavlink.io/en/messages/common.html#MAV_FRAME_LOCAL_OFFSET_NED) \= 7, [MAV_FRAME_BODY_NED](https://mavlink.io/en/messages/common.html#MAV_FRAME_BODY_NED) \= 8, [MAV_FRAME_BODY_OFFSET_NED](https://mavlink.io/en/messages/common.html#MAV_FRAME_BODY_OFFSET_NED) \= 9 |
| type_mask        | uint16_t |       | [POSITION_TARGET_TYPEMASK](https://mavlink.io/en/messages/common.html#POSITION_TARGET_TYPEMASK) | Bitmap to indicate which dimensions should be ignored by the vehicle.                                                                                                                                                                                                                                                                                                                                                     |
| x                | float    | m     |                                                                                                 | X Position in NED frame                                                                                                                                                                                                                                                                                                                                                                                                   |
| y                | float    | m     |                                                                                                 | Y Position in NED frame                                                                                                                                                                                                                                                                                                                                                                                                   |
| z                | float    | m     |                                                                                                 | Z Position in NED frame (note, altitude is negative in NED)                                                                                                                                                                                                                                                                                                                                                               |
| vx               | float    | m/s   |                                                                                                 | X velocity in NED frame                                                                                                                                                                                                                                                                                                                                                                                                   |
| vy               | float    | m/s   |                                                                                                 | Y velocity in NED frame                                                                                                                                                                                                                                                                                                                                                                                                   |
| vz               | float    | m/s   |                                                                                                 | Z velocity in NED frame                                                                                                                                                                                                                                                                                                                                                                                                   |
| afx              | float    | m/s/s |                                                                                                 | X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N                                                                                                                                                                                                                                                                                                                                  |
| afy              | float    | m/s/s |                                                                                                 | Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N                                                                                                                                                                                                                                                                                                                                  |
| afz              | float    | m/s/s |                                                                                                 | Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N                                                                                                                                                                                                                                                                                                                                  |
| yaw              | float    | rad   |                                                                                                 | yaw setpoint                                                                                                                                                                                                                                                                                                                                                                                                              |
| yaw_rate         | float    | rad/s |                                                                                                 | yaw rate setpoint                                                                                                                                                                                                                                                                                                                                                                                                         |

**6\. SET_ATTITUDE_TARGET (82)**

Sets a desired vehicle attitude. Used by an external controller to command the vehicle (manual controller or other system).

| Field Name                                                                         | Type       | Units | Values                                                                                          | Description                                                                                                                                                                                                                                            |
| :--------------------------------------------------------------------------------- | :--------- | :---- | :---------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| time_boot_ms                                                                       | uint32_t   | ms    |                                                                                                 | Timestamp (time since system boot).                                                                                                                                                                                                                    |
| target_system                                                                      | uint8_t    |       |                                                                                                 | System ID                                                                                                                                                                                                                                              |
| target_component                                                                   | uint8_t    |       |                                                                                                 | Component ID                                                                                                                                                                                                                                           |
| type_mask                                                                          | uint8_t    |       | [ATTITUDE_TARGET_TYPEMASK](https://mavlink.io/en/messages/common.html#ATTITUDE_TARGET_TYPEMASK) | Bitmap to indicate which dimensions should be ignored by the vehicle.                                                                                                                                                                                  |
| q                                                                                  | float\[4\] |       |                                                                                                 | Attitude quaternion (w, x, y, z order, zero-rotation is 1, 0, 0, 0\) from [MAV_FRAME_LOCAL_NED](https://mavlink.io/en/messages/common.html#MAV_FRAME_LOCAL_NED) to [MAV_FRAME_BODY_FRD](https://mavlink.io/en/messages/common.html#MAV_FRAME_BODY_FRD) |
| body_roll_rate                                                                     | float      | rad/s |                                                                                                 | Body roll rate                                                                                                                                                                                                                                         |
| body_pitch_rate                                                                    | float      | rad/s |                                                                                                 | Body pitch rate                                                                                                                                                                                                                                        |
| body_yaw_rate                                                                      | float      | rad/s |                                                                                                 | Body yaw rate                                                                                                                                                                                                                                          |
| thrust                                                                             | float      |       |                                                                                                 | Collective thrust, normalized to 0 .. 1 (-1 .. 1 for vehicles capable of reverse thrust)                                                                                                                                                               |
| thrust_body [\++](https://mavlink.io/en/messages/common.html#mav2_extension_field) | float\[3\] |       |                                                                                                 | 3D thrust setpoint in the body NED frame, normalized to \-1 .. 1                                                                                                                                                                                       |

# Vision Stream

**The camera stream is NOT transmitted through MAVLink.** It uses a separate UDP connection.

Frequency: 30 Hz

Resolution: 640 × 360

Image format: JPEG

**UDP Vision Stream for the camera:**

Protocol:UDP

Port: 5600

Byte order: Little Endian

Header size: 24 bytes

Packet structure:

uint32 frame_id  
uint16 chunk_id  
uint16 total_chunks  
uint32 jpeg_size  
uint32 payload_size  
uint64 sim_time_ns

followed by:

payload_size bytes of JPEG data

Vision packets should be grouped by frame_id, reassembled in ascending chunk_id order, duplicate chunks ignored, and any frame with missing chunks or invalid JPEG data discarded in its entirety.

## Software-in-the-Loop Bridge

The simulator provides a low-latency UDP SITL bridge enabling external AI controllers to exchange telemetry and control commands.

## Coordinate Frames

**MAVLink Coordinate Convention**

All coordinates follow the MAVLink 2 North-East-Down (NED) convention:

- **X-axis:** North / Forward
- **Y-axis:** East / Right
- **Z-axis:** Down

**MAV_FRAME_LOCAL_NED**

The local NED frame is a world-fixed coordinate system whose origin `(0,0,0)` is a fixed point on the ground, usually corresponding to the vehicle's arming or takeoff location.

**MAV_FRAME_BODY_NED**

The body frame is attached to the vehicle and moves with it. The origin is located at the vehicle center.

Axis definitions:

- **X-axis:** Forward
- **Y-axis:** Right
- **Z-axis:** Down

**Body to Camera:**

The camera and the body frame have the same origin. The camera is tilted upwards by 20°. Be aware that all coordinates are NED and you might need to rotate the camera frame into the camera coordinate convention of your specific image processing library.

**Body to IMU:**

The body to imu transformation is the identity map.
